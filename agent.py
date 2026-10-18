"""
LangGraph agent: natural-language question -> (pick a collection) -> MongoDB pipeline -> results -> answer.

Multi-collection mode (when catalog.json exists, built by build_catalog.py):

  keywords -> retrieve -> choose -> generate -> validate -> execute -> answer
                                       ^           |          |
                                       +-----------+----------+   (retry with the error, max MAX_ATTEMPTS)

Single-collection mode (no catalog.json): the collection named by COLLECTION_NAME in .env is used.
"""
from __future__ import annotations

import difflib
import json
import math
import operator
import os
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from bson import json_util
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import PyMongoError

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))
MAX_TIME_MS = 8000          # server-side timeout per query
DEFAULT_LIMIT = 100         # appended if the LLM forgets a $limit
MAX_RESULT_CHARS = 30000    # how much result text the answer LLM sees
ROUTER_TOP_K = int(os.getenv("ROUTER_TOP_K", "5"))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBED_DIMS = int(os.getenv("EMBEDDING_DIMS", "768"))
SCHEMA_SAMPLE_SIZE = int(os.getenv("SCHEMA_SAMPLE_SIZE", "500"))
CATALOG_PATH = Path(os.getenv("CATALOG_PATH", str(HERE / "catalog.json")))
DESCRIPTIONS_PATH = HERE / "collection_descriptions.json"   # optional: {"orders": "One doc per purchase", ...}
NOTES_DIR = HERE / "notes"                                  # optional: notes/<collection>.txt

ALLOWED_STAGES = {
    "$match", "$group", "$project", "$sort", "$limit", "$skip", "$unwind", "$count",
    "$addFields", "$set", "$unset", "$facet", "$bucket", "$bucketAuto",
    "$sortByCount", "$replaceRoot", "$replaceWith", "$sample",
}
BLOCKED_OPERATORS = {"$out", "$merge", "$where", "$function", "$accumulator", "$lookup", "$graphLookup", "$unionWith"}
RESHAPING_STAGES = {
    "$group", "$project", "$addFields", "$set", "$unset", "$facet", "$bucket",
    "$bucketAuto", "$sortByCount", "$replaceRoot", "$replaceWith", "$count", "$unwind",
}


class QueryError(Exception):
    """An error whose message is written to be read by the LLM."""


# ----------------------------------------------------------------------------
# Resources (lazy, so importing this module never needs credentials)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_client():
    missing = [k for k in ("MONGODB_URI", "DB_NAME") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing {missing} in {HERE / '.env'}")
    return MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=8000)


def get_db():
    return get_client()[os.environ["DB_NAME"]]


def get_collection(name: Optional[str] = None):
    name = name or os.getenv("COLLECTION_NAME")
    if not name:
        raise RuntimeError("No collection given and COLLECTION_NAME is not set in .env")
    return get_db()[name]


def _model_names() -> list:
    primary = os.getenv("LLM_MODEL", "gemini-3.6-flash")
    fallbacks = [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "gemini-3.5-flash,gemini-2.5-flash").split(",") if m.strip()]
    return [primary] + [m for m in fallbacks if m != primary]


@lru_cache(maxsize=1)
def get_models() -> tuple:
    """Primary model first, then fallbacks. Temperature is left at the default on purpose (Gemini 3)."""
    return tuple(ChatGoogleGenerativeAI(model=name, max_retries=4) for name in _model_names())


def get_llm():
    models = get_models()
    return models[0].with_fallbacks(list(models[1:])) if len(models) > 1 else models[0]


def get_structured_llm(schema):
    chains = [m.with_structured_output(schema) for m in get_models()]
    return chains[0].with_fallbacks(chains[1:]) if len(chains) > 1 else chains[0]


def _text_of(content: Any) -> str:
    """Gemini may return a string or a list of content parts; normalise to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "") for part in content if isinstance(part, (str, dict))
        )
    return str(content)


def make_embedder(model: str = EMBED_MODEL, dims: int = EMBED_DIMS):
    return GoogleGenerativeAIEmbeddings(model=model, output_dimensionality=dims)


def with_retries(fn, tries: int = 4, base: float = 2.0):
    """Retry on temporary API errors (overloaded / rate limited) with exponential backoff."""
    transient = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "TIMEOUT", "TIMED OUT", "DEADLINE")
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1 or not any(m in str(e).upper() for m in transient):
                raise
            time.sleep(base * (2 ** i))


# ----------------------------------------------------------------------------
# Schema inference (sampling a collection)
# ----------------------------------------------------------------------------
MAX_ENUM_VALUES = 15        # string fields with <= this many distinct values are listed in full
CARD_ENUM_VALUES = 8        # ... and in the compact "card" used for routing
CARD_MAX_CHARS = 3500       # keeps embedding input well under the model limit


def _type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "double"
    if isinstance(v, str):
        return "string"
    if isinstance(v, datetime):
        return "date"
    if isinstance(v, dict):
        return "object"
    if isinstance(v, (list, tuple)):
        return "array"
    return type(v).__name__.lower()          # ObjectId -> "objectid", Decimal128 -> "decimal128"


def sample_field_stats(coll, sample_size: int = SCHEMA_SAMPLE_SIZE) -> tuple:
    """Sample documents; return ({field_path: stats}, number_of_docs_sampled)."""
    docs = list(coll.aggregate([{"$sample": {"size": sample_size}}], maxTimeMS=20000))
    if not docs:
        raise RuntimeError("The collection is empty (or the user cannot read it).")
    stats: dict = {}

    def entry(path):
        return stats.setdefault(path, {"docs": 0, "types": {}, "values": {}, "examples": []})

    def walk(val, path, seen):
        if isinstance(val, dict):
            if path:
                seen.add(path)
                e = entry(path); e["types"]["object"] = e["types"].get("object", 0) + 1
            for k, v in val.items():
                walk(v, f"{path}.{k}" if path else str(k), seen)
        elif isinstance(val, (list, tuple)):
            seen.add(path)
            e = entry(path); e["types"]["array"] = e["types"].get("array", 0) + 1
            for item in val:                 # array elements share the array's path
                walk(item, path, seen)
        else:
            seen.add(path)
            e = entry(path)
            t = _type_name(val)
            e["types"][t] = e["types"].get(t, 0) + 1
            if isinstance(val, str):
                if len(e["values"]) <= 200:
                    e["values"][val] = e["values"].get(val, 0) + 1
            elif len(e["examples"]) < 3 and t != "null":
                e["examples"].append(str(val)[:40])

    for d in docs:
        seen: set = set()
        walk(d, "", seen)
        for path in seen:
            stats[path]["docs"] += 1
    return stats, len(docs)


def render_schema_text(stats: dict, n: int) -> str:
    """Detailed schema description used in the query-generation prompt."""
    lines = [f"Based on a random sample of {n} documents (fields not listed here were not seen):"]
    for path in sorted(stats):
        e = stats[path]
        types = " | ".join(sorted(e["types"], key=lambda t: -e["types"][t]))
        line = f"- {path}: {types}, present in {round(100 * e['docs'] / n)}% of docs"
        vals = e["values"]
        if vals and len(vals) <= MAX_ENUM_VALUES:
            line += f"; allowed values seen: {sorted(vals, key=lambda v: -vals[v])}"
        elif vals:
            top = sorted(vals, key=lambda v: -vals[v])[:3]
            line += f"; examples: {[v[:40] for v in top]}"
        elif e["examples"]:
            line += f"; examples: {e['examples']}"
        lines.append(line)
    lines.append("(For fields that are arrays, the listed types describe the array and its elements; "
                 "sub-fields of array elements use dot notation and need $unwind.)")
    return "\n".join(lines)


def render_card(name: str, doc_count: int, stats: dict, description: str = "") -> str:
    """Compact one-paragraph summary of a collection. This is what gets embedded and shown to the router."""
    lines = [f"Collection: {name} (~{doc_count:,} documents)"]
    if description:
        lines.append(f"Description: {description}")
    parts = []
    for path in sorted(stats):
        e = stats[path]
        types = " | ".join(sorted(e["types"], key=lambda t: -e["types"][t]))
        item = f"{path} ({types}"
        vals = e["values"]
        if vals and len(vals) <= CARD_ENUM_VALUES:
            item += ": " + " | ".join(v[:20] for v in sorted(vals, key=lambda v: -vals[v]))
        parts.append(item + ")")
    fields = "Fields: " + ", ".join(parts)
    if len(fields) > CARD_MAX_CHARS:
        fields = fields[:CARD_MAX_CHARS].rsplit(",", 1)[0] + ", ..."
    lines.append(fields)
    return "\n".join(lines)


def infer_schema(coll, sample_size: int = SCHEMA_SAMPLE_SIZE) -> tuple:
    stats, n = sample_field_stats(coll, sample_size)
    return render_schema_text(stats, n), frozenset(stats)


# ----------------------------------------------------------------------------
# Schema from a schema.json file (single-collection mode only)
# ----------------------------------------------------------------------------
def _flatten_json_schema(node: dict, prefix: str, out: set) -> None:
    for key, sub in (node.get("properties") or {}).items():
        path = f"{prefix}{key}"
        out.add(path)
        if isinstance(sub, dict):
            _descend(sub, path + ".", out)


def _descend(sub: dict, prefix: str, out: set) -> None:
    _flatten_json_schema(sub, prefix, out)
    items = sub.get("items")
    if isinstance(items, dict):
        _descend(items, prefix, out)
    for alt in sub.get("anyOf", []) or []:
        if isinstance(alt, dict):
            _descend(alt, prefix, out)


def _collect_paths(node: Any, out: set) -> None:
    """Generic walker for mongodb-schema style output (objects carrying a 'path')."""
    if isinstance(node, dict):
        path = node.get("path")
        if isinstance(path, list):
            path = ".".join(map(str, path))
        if isinstance(path, str) and path:
            out.add(path)
        for v in node.values():
            _collect_paths(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_paths(v, out)


def _find_json_schema_root(node: Any) -> Optional[dict]:
    if isinstance(node, dict):
        if isinstance(node.get("$jsonSchema"), dict):
            return node["$jsonSchema"]
        if isinstance(node.get("properties"), dict):
            return node
        for v in node.values():
            found = _find_json_schema_root(v)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _find_json_schema_root(v)
            if found:
                return found
    return None


def _fields_from_file() -> frozenset:
    try:
        data = json.loads((HERE / "schema.json").read_text())
    except Exception:
        return frozenset()
    out: set = set()
    root = _find_json_schema_root(data)
    if root:
        _descend(root, "", out)
    if not out:
        _collect_paths(data, out)
    return frozenset(out)


@lru_cache(maxsize=8)
def _single_collection_schema(name: str) -> tuple:
    """(prompt text, field paths) for single-collection mode. SCHEMA_SOURCE=auto|file."""
    source = os.getenv("SCHEMA_SOURCE", "").strip().lower()
    path = HERE / "schema.json"
    use_auto = source == "auto" or (source != "file" and not path.exists())
    if use_auto:
        body, fields = infer_schema(get_collection(name))
        text = "## Schema (auto-detected from the live collection)\n" + body
    else:
        text = "## Schema (generated from Atlas)\n" + path.read_text()
        fields = _fields_from_file()
    notes = HERE / "schema_notes.txt"
    if notes.exists():
        text += "\n\n## Notes about the data\n" + notes.read_text()
    return text, fields


# ----------------------------------------------------------------------------
# Catalog (built once by build_catalog.py)
# ----------------------------------------------------------------------------
def _normalize(vec: list) -> list:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


@lru_cache(maxsize=1)
def get_catalog() -> Optional[dict]:
    """Loaded catalog, or None if catalog.json does not exist (=> single-collection mode)."""
    if not CATALOG_PATH.exists():
        return None
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    for c in data["collections"]:
        c["_vec"] = _normalize(c["embedding"])
    data["by_name"] = {c["name"]: c for c in data["collections"]}
    return data


@lru_cache(maxsize=1)
def get_embedder():
    cat = get_catalog()
    return make_embedder(cat.get("embedding_model", EMBED_MODEL), cat.get("embedding_dims", EMBED_DIMS))


def _notes_for(name: str) -> str:
    f = NOTES_DIR / (re.sub(r"[^\w.-]", "_", name) + ".txt")
    return f.read_text(encoding="utf-8") if f.exists() else ""


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
class State(TypedDict, total=False):
    question: str
    keywords: list               # from the keyword-extraction step
    candidates: list             # [{"name":..., "score":...}] from retrieval
    collection: str              # the chosen collection
    route_reason: str
    schema_text: str             # schema of the chosen collection (prompt text)
    known_fields: list           # field paths of the chosen collection (for validation)
    raw_pipeline: str
    reasoning: str
    pipeline: Optional[list]
    error: Optional[str]
    result: Optional[list]
    attempts: int
    empty_retried: bool
    answer: str
    trace: Annotated[list, operator.add]   # step-by-step log shown in the web UI


# ----------------------------------------------------------------------------
# Routing nodes: keywords -> retrieve -> choose
# ----------------------------------------------------------------------------
class Keywords(BaseModel):
    keywords: list[str] = Field(description="Meaningful search keywords only, lowercase, no duplicates, at most 12.")


KEYWORD_PROMPT = """Extract search keywords from a question that will be asked of a database.

Remove filler: stop words, question words (how, what, which, who, show, list, give, tell, find), articles,
prepositions, pronouns and politeness.
Keep: nouns that name the data (orders, customers, restaurants), attribute names (price, city, status),
specific values (names, places, categories), numbers, and date or time words.
Return the keywords lowercase, without duplicates, at most 12."""


def keywords_node(state: State) -> dict:
    out: Keywords = get_structured_llm(Keywords).invoke(
        [SystemMessage(content=KEYWORD_PROMPT), HumanMessage(content=state["question"])]
    )
    seen, kws = set(), []
    for k in out.keywords:
        k = k.strip().lower()
        if k and k not in seen:
            seen.add(k); kws.append(k)
    kws = kws[:12] or [state["question"].strip().lower()]    # never search with nothing
    return {"keywords": kws, "trace": [{"step": "keywords", "keywords": kws}]}


def retrieve_node(state: State) -> dict:
    cat = get_catalog()
    query = ", ".join(state["keywords"])
    qv = _normalize(with_retries(lambda: get_embedder().embed_query(query)))
    scored = []
    for c in cat["collections"]:
        if len(c["_vec"]) != len(qv):
            raise RuntimeError("Embedding size differs from the catalog; re-run build_catalog.py with the same EMBEDDING_* settings.")
        scored.append((sum(a * b for a, b in zip(qv, c["_vec"])), c["name"]))
    scored.sort(reverse=True)
    cands = [{"name": n, "score": round(s, 3)} for s, n in scored[:ROUTER_TOP_K]]
    return {"candidates": cands, "trace": [{"step": "retrieve", "candidates": cands}]}


class Choice(BaseModel):
    collection: str = Field(description="The exact name of the single best collection from the candidates.")
    reason: str = Field(description="One short sentence explaining why.")


CHOOSE_PROMPT = """You route a question to exactly ONE MongoDB collection.

The candidates below were retrieved by text similarity, which can be wrong, so read each one.
Choose the collection whose documents actually contain the data needed to answer the question.
Prefer the collection that has the fields the question asks about. Return its exact name.

Candidates:
{candidates}"""


def _match_name(raw: str, names: list) -> Optional[str]:
    raw = (raw or "").strip().strip("`'\"")
    for n in names:
        if n == raw:
            return n
    for n in names:
        if n.lower() == raw.lower():
            return n
    return None


def choose_node(state: State) -> dict:
    cat = get_catalog()
    names = [c["name"] for c in state["candidates"]]
    if len(names) == 1:
        name, reason = names[0], "Only one candidate."
    else:
        block = "\n\n".join(f"### {n}\n{cat['by_name'][n]['card']}" for n in names)
        out: Choice = get_structured_llm(Choice).invoke([
            SystemMessage(content=CHOOSE_PROMPT.format(candidates=block)),
            HumanMessage(content=f"Question: {state['question']}"),
        ])
        name = _match_name(out.collection, names)
        reason = out.reason
        if name is None:
            name = names[0]
            reason = f"The model returned an unknown collection ('{out.collection}'); using the top retrieval hit."
    entry = cat["by_name"][name]
    text = f"## Collection: {name}\n" + entry["schema_text"]
    notes = _notes_for(name)
    if notes:
        text += "\n\n## Notes about the data\n" + notes
    return {
        "collection": name, "route_reason": reason, "schema_text": text, "known_fields": list(entry["fields"]),
        "trace": [{"step": "choose", "collection": name, "reason": reason}],
    }


def default_collection_node(state: State) -> dict:
    """Single-collection mode: no catalog, use COLLECTION_NAME."""
    name = os.environ["COLLECTION_NAME"]
    text, fields = _single_collection_schema(name)
    return {
        "collection": name, "schema_text": text, "known_fields": sorted(fields), "keywords": [], "candidates": [],
        "route_reason": "Single-collection mode.",
        "trace": [{"step": "use_default", "collection": name}],
    }


# ----------------------------------------------------------------------------
# generate
# ----------------------------------------------------------------------------
class MongoQuery(BaseModel):
    reasoning: str = Field(description="One or two sentences: how this pipeline answers the question.")
    pipeline_json: str = Field(
        description=(
            "A JSON array (as a string) of MongoDB aggregation stages. "
            'Dates must use extended JSON, e.g. {"$date": "2026-01-01T00:00:00Z"}. '
            "No comments, no markdown fences."
        )
    )


SYSTEM_PROMPT = """You convert natural-language questions into MongoDB aggregation pipelines.

Rules:
- Query ONLY the single collection described below. Never use $lookup, $out, $merge, $where or $function.
- Return a pipeline as a JSON array of stages.
- Use only field names that exist in the schema. Nested fields use dot notation ("customer.city"); to use array-of-object fields, $unwind first.
- Respect the exact value spellings and casing described in the schema and notes.
- Dates: use extended JSON like {{"$date": "2026-01-01T00:00:00Z"}}. Today's date is {today}.
- Always end with a $limit (<= 100) unless the pipeline already produces one small result (e.g. $group into a single doc or $count).
- Prefer $match early so indexes can be used. Do NOT add stages the question doesn't need.

{schema}
"""


def generate_node(state: State) -> dict:
    system = SYSTEM_PROMPT.format(schema=state["schema_text"], today=datetime.now(timezone.utc).date().isoformat())
    user = f"Question: {state['question']}"
    if state.get("error"):
        user += (
            f"\n\nYour previous pipeline:\n{state.get('raw_pipeline', '')}\n\n"
            f"It had this problem:\n{state['error']}\n\n"
            "Return a corrected pipeline. If the problem says a filter matched 0 documents, "
            "fix the filter values/fields using the information given."
        )
    out: MongoQuery = get_structured_llm(MongoQuery).invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return {
        "raw_pipeline": out.pipeline_json,
        "reasoning": out.reasoning,
        "attempts": state.get("attempts", 0) + 1,
        "error": None,
        "trace": [{
            "step": "generate", "attempt": state.get("attempts", 0) + 1,
            "pipeline": out.pipeline_json, "reasoning": out.reasoning,
        }],
    }


# ----------------------------------------------------------------------------
# validate (pure code, no DB call)
# ----------------------------------------------------------------------------
def parse_pipeline(raw: str) -> list:
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        pipeline = json_util.loads(text)  # understands {"$date": ...}, {"$oid": ...}
    except Exception as e:
        raise QueryError(f"Pipeline is not valid JSON ({e}). Return a JSON array of stages only.")
    if not isinstance(pipeline, list) or not pipeline:
        raise QueryError("Pipeline must be a non-empty JSON array of stage objects.")
    return pipeline


def _find_operators(node: Any, found: set) -> set:
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.startswith("$"):
                found.add(k)
            _find_operators(v, found)
    elif isinstance(node, list):
        for v in node:
            _find_operators(v, found)
    return found


def _match_fields(cond: Any, out: set) -> None:
    if not isinstance(cond, dict):
        return
    for k, v in cond.items():
        if k in ("$and", "$or", "$nor") and isinstance(v, list):
            for sub in v:
                _match_fields(sub, out)
        elif not k.startswith("$"):
            out.add(k)


def _is_known(field: str, known: frozenset) -> bool:
    return (
        field in known
        or any(k.startswith(field + ".") for k in known)
        or any(field.startswith(k + ".") for k in known)
    )


def _suggest(field: str, known: frozenset) -> list:
    return difflib.get_close_matches(field, sorted(known), n=3, cutoff=0.4)


def static_check(pipeline: list, known: frozenset) -> list:
    for i, stage in enumerate(pipeline):
        if not isinstance(stage, dict) or len(stage) != 1:
            raise QueryError(f"Stage {i} must be an object with exactly one operator key, got: {stage}")
        name = next(iter(stage))
        if name not in ALLOWED_STAGES:
            raise QueryError(f"Stage {name} is not allowed. Allowed stages: {sorted(ALLOWED_STAGES)}")

    bad = _find_operators(pipeline, set()) & BLOCKED_OPERATORS
    if bad:
        raise QueryError(f"Operators not allowed: {sorted(bad)}. Only the single collection can be queried.")

    if known:
        for stage in pipeline:
            name = next(iter(stage))
            if name in RESHAPING_STAGES:
                break
            fields: set = set()
            if name == "$match":
                _match_fields(stage["$match"], fields)
            elif name == "$sort" and isinstance(stage["$sort"], dict):
                fields = set(stage["$sort"])
            for f in fields:
                if not _is_known(f, known):
                    raise QueryError(
                        f"Field '{f}' does not exist in the collection. "
                        f"Closest existing fields: {_suggest(f, known)}"
                    )

    if "$limit" not in pipeline[-1]:
        pipeline = pipeline + [{"$limit": DEFAULT_LIMIT}]
    return pipeline


def validate_node(state: State) -> dict:
    try:
        pipeline = static_check(parse_pipeline(state["raw_pipeline"]), frozenset(state.get("known_fields") or []))
    except QueryError as e:
        return {"pipeline": None, "error": str(e), "trace": [{"step": "validate", "ok": False, "message": str(e)}]}
    return {"pipeline": pipeline, "error": None, "trace": [{"step": "validate", "ok": True}]}


# ----------------------------------------------------------------------------
# execute against Atlas (explain first, then run)
# ----------------------------------------------------------------------------
def diagnose_empty(coll, pipeline: list, max_values: int = 15) -> str:
    """Test each top-level $match condition alone to find which one kills the result."""
    first = pipeline[0] if pipeline else {}
    if "$match" not in first or not isinstance(first["$match"], dict):
        return "The query ran but returned no documents. Re-check the filters."
    notes = []
    for field, cond in first["$match"].items():
        if field.startswith("$"):
            continue
        try:
            if coll.count_documents({field: cond}, maxTimeMS=4000) == 0:
                vals = [
                    d["_id"]
                    for d in coll.aggregate(
                        [{"$group": {"_id": f"${field}"}}, {"$limit": max_values}], maxTimeMS=4000
                    )
                ]
                notes.append(
                    f"Condition {{{field}: {cond}}} matches 0 documents. "
                    f"Example actual values of '{field}': {vals}"
                )
        except PyMongoError:
            continue
    return " ".join(notes) or (
        "Each filter matches on its own but the combination returns nothing. "
        "Check for conflicting conditions or wrong date ranges."
    )


def execute_node(state: State) -> dict:
    coll = get_collection(state["collection"])
    pipeline = state["pipeline"]
    try:
        coll.database.command(
            {"explain": {"aggregate": coll.name, "pipeline": pipeline, "cursor": {}}, "verbosity": "queryPlanner"}
        )
        docs = list(coll.aggregate(pipeline, maxTimeMS=MAX_TIME_MS))
    except PyMongoError as e:
        details = getattr(e, "details", None) or {}
        msg = details.get("errmsg") or str(e)
        err = f"MongoDB rejected or failed the pipeline: {msg}"
        return {"result": None, "error": err, "trace": [{"step": "execute", "ok": False, "message": err}]}

    if not docs and not state.get("empty_retried"):
        err = diagnose_empty(coll, pipeline)
        return {"result": None, "error": err, "empty_retried": True,
                "trace": [{"step": "execute", "ok": False, "message": "No documents returned. " + err}]}
    return {"result": docs, "error": None, "trace": [{"step": "execute", "ok": True, "count": len(docs)}]}


# ----------------------------------------------------------------------------
# answer / give up
# ----------------------------------------------------------------------------
def answer_node(state: State) -> dict:
    docs = state.get("result") or []
    text = json.dumps(docs, default=str)
    truncated = len(text) > MAX_RESULT_CHARS
    if truncated:
        text = text[:MAX_RESULT_CHARS]
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Collection queried: {state.get('collection')}\n"
        f"MongoDB pipeline that was run:\n{json.dumps(state['pipeline'], default=str)}\n\n"
        f"Results ({len(docs)} documents{', output truncated' if truncated else ''}):\n{text}\n\n"
        "Answer the question using ONLY these results. Be concise. If there are no results, say no matching "
        "data was found. If the result count equals a $limit in the pipeline, mention the list may be capped."
    )
    reply = get_llm().invoke([HumanMessage(content=prompt)])
    return {"answer": _text_of(reply.content)}


def give_up_node(state: State) -> dict:
    return {
        "answer": (
            f"Sorry, I couldn't produce a working query after {state.get('attempts', 0)} attempts.\n"
            f"Last problem: {state.get('error')}\nTry rephrasing the question."
        )
    }


# ----------------------------------------------------------------------------
# Routing + graph
# ----------------------------------------------------------------------------
def _retry_or_stop(state: State, ok_target: str) -> str:
    if not state.get("error"):
        return ok_target
    return "generate" if state.get("attempts", 0) < MAX_ATTEMPTS else "give_up"


def route_after_validate(state: State) -> str:
    return _retry_or_stop(state, "execute")


def route_after_execute(state: State) -> str:
    return _retry_or_stop(state, "answer")


def route_entry(state: State) -> str:
    return "keywords" if get_catalog() else "default_collection"


def build_graph():
    g = StateGraph(State)
    g.add_node("keywords", keywords_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("choose", choose_node)
    g.add_node("default_collection", default_collection_node)
    g.add_node("generate", generate_node)
    g.add_node("validate", validate_node)
    g.add_node("execute", execute_node)
    g.add_node("answer", answer_node)
    g.add_node("give_up", give_up_node)

    g.add_conditional_edges(START, route_entry, {"keywords": "keywords", "default_collection": "default_collection"})
    g.add_edge("keywords", "retrieve")
    g.add_edge("retrieve", "choose")
    g.add_edge("choose", "generate")
    g.add_edge("default_collection", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", route_after_validate, {"execute": "execute", "generate": "generate", "give_up": "give_up"})
    g.add_conditional_edges("execute", route_after_execute, {"answer": "answer", "generate": "generate", "give_up": "give_up"})
    g.add_edge("answer", END)
    g.add_edge("give_up", END)
    return g.compile()


def initial_state(question: str) -> dict:
    return {"question": question.strip(), "attempts": 0, "empty_retried": False, "trace": []}


def ask(question: str, app=None) -> State:
    return (app or build_graph()).invoke(initial_state(question))