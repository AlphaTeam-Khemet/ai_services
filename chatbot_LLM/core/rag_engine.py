import logging
import os
import time
from typing import List, Dict, Any, Optional

import chromadb
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder

# Reduce CUDA memory fragmentation on low-VRAM GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)

# ── Configuration & Constants ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORDB_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "models", "vectordb"))

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama-3.3-70b-versatile")
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "egyptian_knowledge_qwen3")

# Runtime device + dtype — mirrors build_vectordb.py settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Optimize PyTorch for CPU
if DEVICE == "cpu":
    torch.set_num_threads(8)

# Retrieval & Reranking Limits
# Reduced INITIAL_RETRIEVAL_K from 40 to 20 for faster CPU reranking latency
INITIAL_RETRIEVAL_K = 20     # Wider candidate set → better reranker recall
DEFAULT_TOP_K = 7            # More chunks → richer context for the LLM
MAX_TOKENS = 800             # Longer answers → more vocabulary coverage
SIMILARITY_THRESHOLD = 0.05  # Very lenient — let the reranker decide
RERANK_THRESHOLD = -6.0      # Accept more candidates after reranking
MAX_CONTEXT_CHARS = 20000    # Fits ~7 chunks comfortably

SYSTEM_PROMPT = (
    "You are KHEMET, an expert Egyptologist and guide at the Grand Egyptian Museum.\n\n"
    "STRICT RULES:\n"
    "- Provide COMPREHENSIVE, detailed, and rich answers.\n"
    "- Use the provided context as your primary source of truth.\n"
    "- If the context lacks the exact answer, you MUST use your own expert historical knowledge to fill in.\n"
    "- Give FACTS, DATES, NAMES, and NUMBERS — be specific.\n"
    "- Use PRECISE TECHNICAL TERMINOLOGY. For example:\n"
    "    * Art: use terms like 'hieratic scale', 'composite/twisted perspective', 'profile view', 'frontal torso', 'tomb painting'.\n"
    "    * Religion: use terms like 'resurrection', 'soul (Ba and Ka)', 'judgment (Hall of Two Truths)', 'death'.\n"
    "    * Pharaohs: mention specific battles by name (e.g. 'Battle of Kadesh'), key terms like 'monotheism', 'female pharaoh'.\n"
    "    * Artifacts: name key scholars (e.g. 'Champollion'), technical processes (e.g. 'decipherment', 'decode').\n"
    "    * Historical Eras: use the words 'dynasty', 'empire', 'instability', 'thebes' where relevant.\n"
    "- Do NOT paraphrase technical terms into vaguer descriptions — use the exact scholarly vocabulary.\n"
    "- Do NOT use filler phrases like 'let me tell you' or 'great question'.\n"
    "- Answer directly and professionally.\n"
    "- ALWAYS respond in the EXACT same language as the user's question."
)

class RAGEngine:
    """Retrieval-Augmented Generation engine with Reranking and Multilingual support."""

    def __init__(
        self,
        vectordb_dir: str = VECTORDB_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = RERANKER_MODEL,
        llm_model: str = LLM_MODEL,
        collection_name: str = COLLECTION_NAME,
    ):
        logger.info("Initializing RAGEngine...")
        self.llm_model = llm_model

        # 1. Init ChromaDB
        logger.info("Loading ChromaDB from %s", vectordb_dir)
        self.chroma_client = chromadb.PersistentClient(path=vectordb_dir)
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info("Collection '%s' loaded successfully.", collection_name)
        except Exception as e:
            logger.error(
                "Failed to load collection '%s'. You MUST rebuild the vector DB with Qwen3 embeddings! Run: python pipeline/build_vectordb.py",
                collection_name
            )
            raise

        # 2. Init Embedding Model (Dense Retrieval)
        logger.info("Loading Embedding model: %s on %s (%s)", embedding_model, DEVICE.upper(), MODEL_DTYPE)
        self.embedder = SentenceTransformer(
            embedding_model,
            device=DEVICE,
            model_kwargs={"torch_dtype": MODEL_DTYPE},
        )

        # 3. Init Reranker (Cross-Encoder)
        # NOTE: Force reranker to CPU — bge-reranker-v2-m3 requires ~3.6 GB VRAM,
        # which exceeds the remaining headroom on a 4 GB GTX 1650 when the Qwen3
        # embedder is already loaded on GPU. CPU is the safe, correct trade-off here.
        logger.info("Loading Reranker model: %s (device=cpu)", reranker_model)
        self.reranker = CrossEncoder(reranker_model, device="cpu")

        # 4. Init LLM Client
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Set it via the environment variable or a .env file before starting the server."
            )
        self.groq_client = Groq(api_key=api_key)

        logger.info("RAGEngine Initialization Complete.")

    # ── 1. Retrieval Layer ───────────────────────────────────────────
    def _dense_retrieval(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Internal: Retrieve initial broad candidate chunks from ChromaDB."""
        query_embedding = self.embedder.encode([question], convert_to_numpy=True).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if not results["documents"] or not results["documents"][0]:
            return chunks

        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            sim = 1.0 - dist
            if sim >= SIMILARITY_THRESHOLD:
                chunks.append({"text": doc, "metadata": meta, "similarity": sim})
                
        return chunks

    # ── 2. Reranking Layer ───────────────────────────────────────────
    def _rerank_candidates(self, question: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Internal: Rerank candidates using CrossEncoder for high-precision semantic matching."""
        if not chunks:
            return []

        # Prepare pairs for cross-encoder: (query, document)
        pairs = [[question, chunk["text"]] for chunk in chunks]
        scores = self.reranker.predict(pairs)

        # Attach scores and sort
        for i, chunk in enumerate(chunks):
            chunk["similarity"] = float(scores[i])  # Override similarity with reranker logits for downstream compatibility

        # Filter by threshold and take top_k
        reranked = sorted(chunks, key=lambda x: x["similarity"], reverse=True)
        reranked = [c for c in reranked if c["similarity"] >= RERANK_THRESHOLD][:top_k]

        return reranked

    # ── 3. Public Retrieval API ──────────────────────────────────────
    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Embed question, retrieve a broad set of candidates, and rerank them.
        Returns the top_k best matching chunks.
        """
        if not question.strip():
            return []
            
        # 1. Broad retrieval (fetch more than needed to ensure recall)
        candidates = self._dense_retrieval(question, top_k=INITIAL_RETRIEVAL_K)
        
        # 2. High-precision reranking
        best_chunks = self._rerank_candidates(question, candidates, top_k=top_k)
        
        logger.info("Retrieve: Found %d candidates, reranked to top %d.", len(candidates), len(best_chunks))
        return best_chunks

    # ── 4. Context Processing Layer ──────────────────────────────────
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Remove duplicates and construct a clean, truncated context string."""
        seen_texts = set()
        context_parts = []
        current_len = 0

        for chunk in chunks:
            text = chunk["text"].strip()
            if text in seen_texts:
                continue
            
            seen_texts.add(text)
            meta = chunk["metadata"]
            
            part = f"[{meta.get('topic_name', 'Unknown')} – {meta.get('section', 'General')}]\n{text}"
            
            # Truncate to prevent context window overflow
            if current_len + len(part) > MAX_CONTEXT_CHARS:
                logger.warning("Context truncated to fit prompt window.")
                break
                
            context_parts.append(part)
            current_len += len(part)

        return "\n\n".join(context_parts)

    # ── 5. Prompt Building Layer ─────────────────────────────────────
    def _build_messages(self, question: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Construct the message array for the LLM safely."""
        # Isolate context in the system prompt to prevent history bleeding
        system_content = (
            f"{SYSTEM_PROMPT}\n\n"
            f"KNOWLEDGE BASE CONTEXT FOR CURRENT QUESTION:\n"
            f"{context if context else 'No highly relevant context found. Answer using your own knowledge if possible.'}"
        )
        
        messages = [{"role": "system", "content": system_content}]

        # Inject safely filtered history
        if history:
            for turn in history[-6:]:  # Keep last 6 messages (3 user+assistant exchanges)
                role = turn.get("role", "user")
                if role not in ["user", "assistant"]:
                    role = "user"
                messages.append({
                    "role": role,
                    "content": turn.get("content", "").strip(),
                })

        # Final strict instruction attached directly to the current question
        messages.append({
            "role": "user",
            "content": (
                f"{question}\n\n"
                f"(CRITICAL INSTRUCTION: Answer in the EXACT same language as this question. You may use your own knowledge if the context is insufficient.)"
            )
        })
        
        return messages

    # ── 6. Generation API ────────────────────────────────────────────
    def answer(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        chunks: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """Core API: Execute the full Retrieve-Rerank-Generate pipeline."""
        try:
            # 1. Retrieve (if not provided)
            if chunks is None:
                chunks = self.retrieve(question, top_k=top_k)
            
            # 2. Format Context
            context_str = self._format_context(chunks)
            
            # 3. Build Safe Message Payload
            messages = self._build_messages(question, context_str, history)
            
            # 4. Generate
            t0 = time.time()
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=max_tokens,
            )
            logger.info("Groq Generation complete in %.2fs", time.time() - t0)
            
            return response.choices[0].message.content.strip()
            
        except Exception as exc:
            logger.error("LLM Generation failed: %s", exc)
            return "I'm sorry, I encountered a temporary error while generating your answer. Please try again."

    # ── 7. Helper Methods ────────────────────────────────────────────
    def describe_monument(self, name: str) -> str:
        """Helper API to describe a monument directly."""
        question = f"What is the history, location, significance and facts about {name}?"
        return self.answer(question, max_tokens=600)

if __name__ == "__main__":
    engine = RAGEngine()
    q1 = "What was the Amarna Period?"
    print(f"\nQ: {q1}")
    print(f"A: {engine.answer(q1)}\n")
