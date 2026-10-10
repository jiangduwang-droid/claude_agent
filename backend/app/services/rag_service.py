import asyncio
import uuid
from typing import Any

import jieba
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession


class RAGService:
    _embedder = None
    _reranker = None
    _reranker_lock = asyncio.Lock()

    def __init__(self, engine, config=None):
        self._engine = engine
        self._config = config

    def _get_embedder(self):
        if RAGService._embedder is None:
            from sentence_transformers import SentenceTransformer
            model_name = "BAAI/bge-small-zh-v1.5"
            if self._config:
                model_name = self._config.rag_config.get("embedding_model", model_name)
            RAGService._embedder = SentenceTransformer(model_name)
        return RAGService._embedder

    def _get_reranker(self):
        if RAGService._reranker is None:
            from sentence_transformers import CrossEncoder
            model_name = "BAAI/bge-rerank-base"
            if self._config:
                model_name = self._config.rag_config.get("rerank_model", model_name)
            RAGService._reranker = CrossEncoder(model_name)
        return RAGService._reranker

    def _tokenize(self, text: str) -> str:
        words = jieba.cut(text)
        return " ".join(words)

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> list[dict]:
        chunk_size = self._config.rag_config.get("chunk_size", chunk_size) if self._config else chunk_size
        overlap = self._config.rag_config.get("chunk_overlap", overlap) if self._config else overlap

        lines = text.split("\n")
        chunks = []
        current_chunk = ""
        heading_path = []

        for line in lines:
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                heading = line.lstrip("#").strip()
                while len(heading_path) >= level:
                    heading_path.pop()
                if level <= len(heading_path):
                    heading_path[level - 1] = heading
                else:
                    heading_path.append(heading)

            if len(current_chunk) + len(line) + 1 > chunk_size and current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "heading_path": " > ".join(heading_path),
                })
                current_chunk = current_chunk[-overlap:] + "\n" + line
            else:
                current_chunk += "\n" + line if current_chunk else line

        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "heading_path": " > ".join(heading_path),
            })

        return chunks

    async def process_document(
        self, doc_id: uuid.UUID, content: str, filename: str = ""
    ) -> int:
        chunks = self._chunk_text(content)
        try:
            embedder = self._get_embedder()
        except Exception as e:
            print(f"[RAG] Embedder unavailable, indexing without vectors: {e}")
            embedder = None

        chunk_data = []
        for i, chunk in enumerate(chunks):
            embedding = None
            if embedder is not None:
                emb = embedder.encode(chunk["content"])
                embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            tokenized = self._tokenize(chunk["content"])
            chunk_data.append({
                "id": uuid.uuid4(),
                "document_id": doc_id,
                "content": chunk["content"],
                "content_tokens": tokenized,
                "heading_path": chunk["heading_path"],
                "chunk_index": i,
                "embedding": embedding,
            })

        async with self._engine.begin() as conn:
            for cd in chunk_data:
                await conn.execute(
                    text(
                        "INSERT INTO document_chunks "
                        "(id, document_id, content, content_tokens, heading_path, chunk_index, embedding) "
                        "VALUES (:id, :doc_id, :content, :tokens, :heading, :idx, CAST(:emb AS vector)"
                    ),
                    {
                        "id": str(cd["id"]),
                        "doc_id": str(cd["document_id"]),
                        "content": cd["content"],
                        "tokens": cd["content_tokens"],
                        "heading": cd["heading_path"],
                        "idx": cd["chunk_index"],
                        "emb": str(cd["embedding"]),
                    },
                )

        return len(chunks)

    async def _bm25_search(self, query: str, kb_ids: list[str] | None, top_k: int = 20) -> list[dict]:
        tokenized_query = self._tokenize(query)
        async with self._engine.begin() as conn:
            sql = text(
                "SELECT dc.id, dc.content, dc.heading_path, dc.document_id, "
                "ts_rank(to_tsvector('simple', dc.content_tokens), "
                "plainto_tsquery('simple', :q)) as rank "
                "FROM document_chunks dc "
                "JOIN documents d ON dc.document_id = d.id "
                "JOIN knowledge_bases kb ON d.kb_id = kb.id "
                "WHERE to_tsvector('simple', dc.content_tokens) @@ plainto_tsquery('simple', :q) "
                + ("AND kb.id = ANY(:kb_ids) " if kb_ids else "")
                + "ORDER BY rank DESC LIMIT :top_k"
            )
            params = {"q": tokenized_query, "top_k": top_k}
            if kb_ids:
                params["kb_ids"] = kb_ids
            result = await conn.execute(sql, params)
            rows = result.fetchall()

        return [
            {
                "content": r[1],
                "heading_path": r[2],
                "document_id": str(r[3]),
                "rank": float(r[4]),
                "source": "bm25",
            }
            for r in rows
        ]

    async def _vector_search(self, query: str, kb_ids: list[str] | None, top_k: int = 20) -> list[dict]:
        embedder = self._get_embedder()
        embedding = embedder.encode(query)
        emb_str = str(embedding.tolist() if hasattr(embedding, "tolist") else list(embedding))

        async with self._engine.begin() as conn:
            sql = text(
                "SELECT dc.id, dc.content, dc.heading_path, dc.document_id, "
                "dc.embedding <=> CAST(:emb AS vector) as distance "
                "FROM document_chunks dc "
                "JOIN documents d ON dc.document_id = d.id "
                "JOIN knowledge_bases kb ON d.kb_id = kb.id "
                + ("WHERE kb.id = ANY(:kb_ids) " if kb_ids else "WHERE 1=1 ")
                + "ORDER BY dc.embedding <=> CAST(:emb AS vector) LIMIT :top_k"
            )
            params = {"emb": emb_str, "top_k": top_k}
            if kb_ids:
                params["kb_ids"] = [str(kb) for kb in kb_ids]
            result = await conn.execute(sql, params)
            rows = result.fetchall()

        return [
            {
                "content": r[1],
                "heading_path": r[2],
                "document_id": str(r[3]),
                "distance": float(r[4]),
                "source": "vector",
            }
            for r in rows
        ]

    def _rrf_fuse(self, bm25_results: list[dict], vector_results: list[dict], k: int = 5) -> list[dict]:
        scores: dict[str, float] = {}
        content_map: dict[str, dict] = {}

        for rank, r in enumerate(bm25_results):
            cid = str(r["document_id"]) + r["content"][:50]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            content_map[cid] = r

        for rank, r in enumerate(vector_results):
            cid = str(r["document_id"]) + r["content"][:50]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            content_map[cid] = r

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [
            {**content_map[cid], "rrf_score": scores[cid]}
            for cid in sorted_ids[:10]
        ]

    async def _rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        try:
            reranker = self._get_reranker()
            pairs = [(query, c["content"]) for c in candidates]
            scores = reranker.predict(pairs)
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return candidates[:top_k]
        except Exception as e:
            print(f"[RAG] Rerank failed, using RRF results: {e}")
            return candidates[:top_k]

    async def search(
        self,
        query: str,
        user_id: str,
        kb_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        bm25_top_k = self._config.rag_config.get("bm25_top_k", 20) if self._config else 20
        vector_top_k = self._config.rag_config.get("vector_top_k", 20) if self._config else 20
        rrf_k = self._config.rag_config.get("rrf_k", 5) if self._config else 5
        rerank_timeout = self._config.rag_config.get("rerank_timeout", 3.0) if self._config else 3.0

        results = await asyncio.gather(
            self._bm25_search(query, kb_ids, bm25_top_k),
            self._vector_search(query, kb_ids, vector_top_k),
            return_exceptions=True,
        )
        bm25_results = results[0] if not isinstance(results[0], BaseException) else []
        vector_results = results[1] if not isinstance(results[1], BaseException) else []
        if isinstance(results[0], BaseException):
            print(f"[RAG] BM25 search failed: {results[0]}")
        if isinstance(results[1], BaseException):
            print(f"[RAG] Vector search failed: {results[1]}")

        fused = self._rrf_fuse(bm25_results, vector_results, rrf_k)
        if not fused:
            return []

        try:
            reranked = await asyncio.wait_for(
                self._rerank(query, fused, top_k),
                timeout=rerank_timeout,
            )
        except asyncio.TimeoutError:
            print("[RAG] Rerank timed out, using RRF results")
            reranked = fused[:top_k]

        async with self._engine.begin() as conn:
            for r in reranked:
                doc_result = await conn.execute(
                    text("SELECT filename FROM documents WHERE id = :did"),
                    {"did": r["document_id"]},
                )
                doc_row = doc_result.fetchone()
                if doc_row:
                    r["filename"] = doc_row[0]

        return reranked
