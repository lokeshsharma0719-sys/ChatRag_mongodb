"""
Build the collection catalog. Run this ONCE (and again whenever your collections change a lot):

    python build_catalog.py

For every collection in the database it
  1. samples documents and works out the fields, types and common values,
  2. writes a compact "card" describing the collection,
  3. embeds the card with Gemini embeddings,
and saves everything to catalog.json, which the chatbot loads at startup.

Optional inputs (edit these to improve routing, then re-run):
  collection_descriptions.json   {"orders": "One document per customer purchase", ...}
  notes/<collection>.txt         extra notes shown to the query-writing LLM (read at runtime, no rebuild needed)

Options:
  --include a,b      only these collections
  --exclude tmp_,bak substrings; skip collections whose name contains any of them
  --sample 500       documents sampled per collection
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import agent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", default="")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--sample", type=int, default=agent.SCHEMA_SAMPLE_SIZE)
    ap.add_argument("--batch", type=int, default=20, help="cards embedded per API call")
    args = ap.parse_args()

    include = {x.strip() for x in args.include.split(",") if x.strip()}
    exclude = [x.strip() for x in args.exclude.split(",") if x.strip()]
    descriptions = {}
    if agent.DESCRIPTIONS_PATH.exists():
        descriptions = json.loads(agent.DESCRIPTIONS_PATH.read_text(encoding="utf-8"))

    db = agent.get_db()
    infos = [i for i in db.list_collections() if i.get("type", "collection") == "collection"]
    names = sorted(i["name"] for i in infos if not i["name"].startswith("system."))
    if include:
        names = [n for n in names if n in include]
    names = [n for n in names if not any(x in n for x in exclude)]
    print(f"Found {len(names)} collections in '{db.name}'.\n")

    entries, skipped = [], []
    for i, name in enumerate(names, 1):
        coll = db[name]
        try:
            count = coll.estimated_document_count()
            if count == 0:
                skipped.append((name, "empty")); print(f"[{i}/{len(names)}] {name}: empty, skipped"); continue
            stats, n = agent.sample_field_stats(coll, args.sample)
        except Exception as e:
            skipped.append((name, f"{type(e).__name__}: {str(e)[:100]}"))
            print(f"[{i}/{len(names)}] {name}: FAILED ({type(e).__name__}), skipped"); continue
        desc = descriptions.get(name, "")
        entries.append({
            "name": name,
            "doc_count": count,
            "description": desc,
            "fields": sorted(stats),
            "schema_text": agent.render_schema_text(stats, n) if not desc else f"Purpose: {desc}\n" + agent.render_schema_text(stats, n),
            "card": agent.render_card(name, count, stats, desc),
        })
        print(f"[{i}/{len(names)}] {name}: {count:,} docs, {len(stats)} fields")

    if not entries:
        print("\nNothing to index. Check the connection, permissions, and --include/--exclude."); return 1

    print(f"\nEmbedding {len(entries)} collection cards with {agent.EMBED_MODEL} ({agent.EMBED_DIMS} dims)...")
    embedder = agent.make_embedder()
    for start in range(0, len(entries), args.batch):
        chunk = entries[start:start + args.batch]
        vectors = agent.with_retries(lambda: embedder.embed_documents([e["card"] for e in chunk]))
        for e, v in zip(chunk, vectors):
            e["embedding"] = v
        print(f"  embedded {min(start + args.batch, len(entries))}/{len(entries)}")

    catalog = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": db.name,
        "embedding_model": agent.EMBED_MODEL,
        "embedding_dims": len(entries[0]["embedding"]),
        "collections": entries,
    }
    tmp = agent.CATALOG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(catalog), encoding="utf-8")
    os.replace(tmp, agent.CATALOG_PATH)      # write-then-rename so a crash never leaves a half-written catalog

    print(f"\nSaved {len(entries)} collections to {agent.CATALOG_PATH}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for n, why in skipped:
            print(f"  - {n}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
