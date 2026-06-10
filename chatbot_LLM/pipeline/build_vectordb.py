#!/usr/bin/env python3
"""
step2_build_vectordb.py
───────────────────────
Loads all_chunks.json, embeds the "searchable" field with
sentence-transformers (Qwen/Qwen3-Embedding-0.6B), and stores everything
in a persistent ChromaDB collection at models/vectordb/.
"""

import json
import os
import sys
import time

import chromadb
import torch
from sentence_transformers import SentenceTransformer

# Reduce CUDA memory fragmentation — recommended for low-VRAM GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CHUNKS_PATH  = os.path.join(ROOT_DIR, "data", "chunks", "all_chunks.json")
VECTORDB_DIR = os.path.join(ROOT_DIR, "models", "vectordb")

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "egyptian_knowledge_qwen3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# float16 on GPU cuts VRAM from ~2.9 GB → ~1.4 GB, leaving room for batch ops
MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE = 32 if DEVICE == "cuda" else 64    # safe batch size for 3-4 GB VRAM
PROGRESS_EVERY = 10      # print dot every N chunks


def main():
    # ── 1. Load chunks ────────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 2 — BUILD VECTOR DATABASE")
    print("=" * 60)

    print(f"\n📂 Loading chunks from {CHUNKS_PATH} ...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Loaded {len(chunks):,} chunks.\n")

    # ── 2. Load embedding model ───────────────────────────────────────
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL} ...")
    print(f"   Device : {DEVICE.upper()} {'(' + torch.cuda.get_device_name(0) + ')' if DEVICE == 'cuda' else '(no GPU found)'}")
    print(f"   Dtype  : {MODEL_DTYPE}")
    t0 = time.time()
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=DEVICE,
        model_kwargs={"torch_dtype": MODEL_DTYPE},
    )
    print(f"   Model loaded in {time.time() - t0:.1f}s "
          f"(dim={model.get_embedding_dimension()})\n")

    # ── 3. Create / reset ChromaDB ────────────────────────────────────
    os.makedirs(VECTORDB_DIR, exist_ok=True)
    print(f"💾 ChromaDB persistent path: {VECTORDB_DIR}")

    client = chromadb.PersistentClient(path=VECTORDB_DIR)

    # Delete the collection if it already exists (fresh build)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"   Deleted existing collection '{COLLECTION_NAME}'.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"   Created collection '{COLLECTION_NAME}'.\n")

    # ── 4. Embed & store in batches ───────────────────────────────────
    total = len(chunks)
    print(f"⏳ Embedding & storing {total:,} chunks (batch_size={BATCH_SIZE}) ...")
    sys.stdout.write("   Progress: ")
    sys.stdout.flush()

    t0 = time.time()
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = chunks[start:end]

        texts = [c["searchable"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "topic_name": c["topic_name"],
                "topic_id": c["topic_id"],
                "category": c["category"],
                "section": c["section"],
                "period": c["period"],
                "location": c["location"],
            }
            for c in batch
        ]
        documents = [c["text"] for c in batch]

        # Embed
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Add to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # Progress indicator — print every PROGRESS_EVERY chunks
        for i in range(start, end):
            if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == total:
                sys.stdout.write(f" [{i + 1}/{total}]")
                sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n   ✓ Done in {elapsed:.1f}s ({total / elapsed:.0f} chunks/sec)\n")

    # ── 5. Verify ─────────────────────────────────────────────────────
    stored_count = collection.count()
    print(f"📊 Total documents in ChromaDB: {stored_count:,}")
    assert stored_count == total, f"Mismatch! expected {total}, got {stored_count}"
    print("   ✓ Count matches.\n")

    # ── 6. Test query ─────────────────────────────────────────────────
    test_query = "ancient Egyptian religion"
    print(f'🔎 Test query: "{test_query}"')
    query_embedding = model.encode([test_query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    print(f"   Top 5 results:\n")
    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]),
        start=1,
    ):
        sim = 1 - dist  # cosine distance → similarity
        print(f"   {rank}. [{sim:.4f}] {meta['topic_name']} → {meta['section']}")
        preview = doc[:120].replace("\n", " ")
        print(f"      {preview}...")
        print(f"      category={meta['category']}  period={meta['period']}")
        print()

    print("=" * 60)
    print("  ✅  Vector database built successfully!")
    print(f"       Path : {VECTORDB_DIR}")
    print(f"       Docs : {stored_count:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
