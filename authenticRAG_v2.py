"""Authentic RAG v2: improved ingestion, retrieval, scoring, and reranking.

This file intentionally keeps the final answer model unchanged:
    qwen/qwen-2.5-72b-instruct via OpenRouter

Examples:
    python authenticRAG_v2.py ingest --corpus ./corpus_input
    python authenticRAG_v2.py search --question "โรคหัดเยอรมันอันตรายอย่างไร"
    python authenticRAG_v2.py batch --questions questions.txt --output results_v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from openai import OpenAI
from opensearchpy import OpenSearch, helpers
from sentence_transformers import CrossEncoder, SentenceTransformer


# ---------------------------------------------------------------------------
# 1) ค่าตั้งต้นของโมเดลและชื่อ index
#
# แยกชื่อ index เป็น v2 เพื่อไม่ให้กระทบข้อมูลของระบบเดิมจากหน้า 22 และ 25
# FINAL_ANSWER_MODEL คงเป็นโมเดลเดิมตามเงื่อนไข ไม่ได้เปลี่ยน LLM ที่ใช้ตอบ
# ---------------------------------------------------------------------------
FINAL_ANSWER_MODEL = "qwen/qwen-2.5-72b-instruct"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
VECTOR_INDEX = "anthropic-v2-vector-index"
LEXICAL_INDEX = "anthropic-v2-lexical-index"


@dataclass
class Chunk:
    """โครงสร้างข้อมูลของหนึ่ง chunk ที่ใช้ร่วมกันตลอด pipeline.

    chunk_id เป็น stable ID จึงยังอ้างอิง chunk เดิมได้แม้รันระบบใหม่
    heading เก็บเส้นทางหัวข้อ Markdown เพื่อรักษาบริบทของเนื้อหา
    context คือคำอธิบายสั้นที่ช่วยให้ retriever เข้าใจตำแหน่งของ chunk
    """

    chunk_id: str
    source: str
    ordinal: int
    heading: str
    content: str
    context: str = ""

    @property
    def retrieval_text(self) -> str:
        """รวมข้อมูลของ chunk เป็นข้อความสำหรับ retrieval."""
        parts = [self.heading, self.context, self.content]
        return "\n".join(part for part in parts if part).strip()


class MarkdownStructureChunker:
    """แบ่ง Markdown ตามหัวข้อ แล้วรวมย่อหน้าพร้อมข้อความ overlap.

    ระบบเดิมแบ่งด้วยขนาด chunk เป็นหลัก ส่วน v2 รักษาโครงสร้างหัวข้อก่อน
    แล้วจึงควบคุมขนาด ทำให้ข้อความจากคนละหัวข้อปะปนกันน้อยลง
    overlap ช่วยไม่ให้ข้อมูลที่อยู่ตรงรอยต่อหายจาก retrieval context
    """

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def __init__(self, target_chars: int = 1200, max_chars: int = 1600, overlap_chars: int = 220):
        """กำหนดขนาดเป้าหมาย ขนาดสูงสุด และ overlap ของ chunk."""
        if not 0 <= overlap_chars < target_chars <= max_chars:
            raise ValueError("Expected 0 <= overlap < target <= max")
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split_file(self, path: Path) -> list[Chunk]:
        """อ่าน Markdown แล้วคืนรายการ chunks พร้อม stable ID."""
        text = path.read_text(encoding="utf-8-sig")
        sections = self._sections(text)
        chunks: list[Chunk] = []
        ordinal = 0
        for heading, body in sections:
            for content in self._window(body):
                stable = hashlib.sha1(f"{path.name}:{ordinal}:{content}".encode("utf-8")).hexdigest()[:18]
                chunks.append(
                    Chunk(
                        chunk_id=f"v2-{stable}",
                        source=path.name,
                        ordinal=ordinal,
                        heading=heading,
                        content=content,
                    )
                )
                ordinal += 1
        return chunks

    def _sections(self, text: str) -> list[tuple[str, str]]:
        """แบ่ง Markdown ตามหัวข้อและสร้าง breadcrumb ของแต่ละ section."""
        breadcrumb: list[str] = []
        current_heading = "Document"
        current_lines: list[str] = []
        sections: list[tuple[str, str]] = []

        def flush() -> None:
            """เพิ่ม section ปัจจุบันเมื่อมีเนื้อหา."""
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_heading, body))

        for line in text.splitlines():
            match = self.heading_pattern.match(line)
            if not match:
                current_lines.append(line)
                continue
            flush()
            current_lines = []
            level = len(match.group(1))
            title = match.group(2).strip()
            breadcrumb[:] = breadcrumb[: level - 1]
            breadcrumb.append(title)
            current_heading = " > ".join(breadcrumb)
        flush()
        return sections or [("Document", text.strip())]

    def _window(self, body: str) -> Iterable[str]:
        """รวมย่อหน้าเป็น chunks ตามขนาดเป้าหมายพร้อม overlap."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        current = ""
        for paragraph in paragraphs:
            pieces = self._split_oversized(paragraph)
            for piece in pieces:
                candidate = f"{current}\n\n{piece}".strip() if current else piece
                if current and len(candidate) > self.target_chars:
                    yield current
                    overlap = self._tail(current)
                    current = f"{overlap}\n\n{piece}".strip()
                else:
                    current = candidate
                if len(current) >= self.max_chars:
                    yield current[: self.max_chars]
                    current = self._tail(current)
        if current:
            yield current

    def _split_oversized(self, paragraph: str) -> list[str]:
        """แบ่งย่อหน้าที่ยาวเกินกำหนด โดยพยายามตัดตรงขอบประโยค."""
        if len(paragraph) <= self.max_chars:
            return [paragraph]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", paragraph) if s.strip()]
        if len(sentences) == 1:
            return [paragraph[i : i + self.max_chars] for i in range(0, len(paragraph), self.max_chars)]
        pieces, current = [], ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > self.max_chars:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces

    def _tail(self, text: str) -> str:
        """เลือกข้อความท้าย chunk เพื่อใช้เป็น overlap."""
        tail = text[-self.overlap_chars :]
        boundary = max(tail.find("\n"), tail.find(" "))
        return tail[boundary + 1 :] if boundary >= 0 else tail


class AuthenticRAGv2:
    """บริการหลักที่ดูแลการ ingest, retrieval, reranking และ answer generation."""

    def __init__(
        self,
        opensearch_host: str = "localhost",
        opensearch_port: int = 9200,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
    ):
        """เชื่อมบริการและโหลด embedding model กับ reranker."""
        # โหลด API key จาก environment เท่านั้น เพื่อไม่ให้ secret อยู่ใน source code
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        self.llm = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        self.search_client = OpenSearch(
            hosts=[{"host": opensearch_host, "port": opensearch_port}],
            use_ssl=False,
        )
        # Bi-encoder ใช้สร้าง vector แยกกัน จึงค้น candidate จำนวนมากได้เร็ว
        self.embedder = SentenceTransformer(embedding_model)

        # Cross-encoder อ่าน question กับ chunk พร้อมกัน แม่นกว่าแต่ช้ากว่า
        # จึงใช้เฉพาะ candidate ที่ผ่าน retrieval มาแล้ว ไม่ใช้กับทุก chunk
        self.reranker = CrossEncoder(reranker_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()

    def create_indices(self, recreate: bool = False) -> None:
        """สร้าง Vector และ Lexical index ของ v2.

        Vector index ใช้ HNSW สำหรับ approximate nearest-neighbor search
        Lexical index ใช้ character n-gram 2-4 ตัวอักษรเพื่อรองรับภาษาไทย
        ที่ไม่มีช่องว่างคั่นทุกคำเหมือนภาษาอังกฤษ
        """
        if recreate:
            for name in (VECTOR_INDEX, LEXICAL_INDEX):
                if self.search_client.indices.exists(index=name):
                    self.search_client.indices.delete(index=name)

        if not self.search_client.indices.exists(index=VECTOR_INDEX):
            # HNSW ลดเวลาค้น vector เมื่อจำนวนเอกสารเพิ่มขึ้น
            self.search_client.indices.create(
                index=VECTOR_INDEX,
                body={
                    "settings": {"index.knn": True},
                    "mappings": {
                        "properties": {
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": self.dimension,
                                "method": {
                                    "name": "hnsw",
                                    "engine": "lucene",
                                    "space_type": "cosinesimil",
                                    "parameters": {"ef_construction": 128, "m": 24},
                                },
                            },
                            "chunk_id": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "ordinal": {"type": "integer"},
                            "heading": {"type": "text"},
                            "content": {"type": "text"},
                            "context": {"type": "text"},
                        }
                    },
                },
            )

        if not self.search_client.indices.exists(index=LEXICAL_INDEX):
            # ใช้ n-gram ทั้งตอน index และ query เพื่อให้คำไทยที่ติดกัน match กันได้
            self.search_client.indices.create(
                index=LEXICAL_INDEX,
                body={
                    "settings": {
                        "index.max_ngram_diff": 2,
                        "analysis": {
                            "tokenizer": {
                                "thai_char_tokenizer": {
                                    "type": "ngram",
                                    "min_gram": 2,
                                    "max_gram": 4,
                                    "token_chars": ["letter", "digit"],
                                }
                            },
                            "analyzer": {
                                "thai_char_index": {
                                    "type": "custom",
                                    "tokenizer": "thai_char_tokenizer",
                                    "filter": ["lowercase"],
                                }
                            },
                        },
                    },
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "ordinal": {"type": "integer"},
                            "heading": {"type": "text", "analyzer": "thai_char_index", "search_analyzer": "thai_char_index"},
                            "content": {"type": "text", "analyzer": "thai_char_index", "search_analyzer": "thai_char_index"},
                            "context": {"type": "text", "analyzer": "thai_char_index", "search_analyzer": "thai_char_index"},
                        }
                    },
                },
            )

    def contextualize(self, chunk: Chunk, document_excerpt: str) -> str:
        """สร้างคำอธิบายสั้นที่บอกว่า chunk นี้อยู่ส่วนใดของเอกสาร.

        ต่างจากระบบเดิมที่ส่งเพียงเนื้อหา chunk เป็นทั้งเอกสารและ chunk
        ฟังก์ชันนี้ส่ง document excerpt และชื่อ section เข้าไปด้วย
        """
        prompt = f"""<document_excerpt>
{document_excerpt[:6000]}
</document_excerpt>
<section>{chunk.heading}</section>
<chunk>{chunk.content}</chunk>

Give a short context that situates this chunk in the document for search retrieval.
Answer only with the context, in the language used by the chunk."""
        response = self.llm.chat.completions.create(
            model=FINAL_ANSWER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()

    def ingest(self, corpus: Path, recreate: bool = False, contextualize: bool = True) -> int:
        """แบ่ง Markdown และบันทึก chunks ลง Vector/Lexical index."""
        # ขั้นตอนที่ 1: สร้าง index ชุด v2 โดยแยกจาก index ของระบบเดิม
        self.create_indices(recreate=recreate)

        # ขั้นตอนที่ 2: อ่าน Markdown และแบ่งเอกสารตามโครงสร้างหัวข้อ
        chunker = MarkdownStructureChunker()
        chunks: list[Chunk] = []
        for path in sorted(corpus.glob("*.md")):
            file_chunks = chunker.split_file(path)

            # ขั้นตอนที่ 3: สร้าง context ให้แต่ละ chunk โดยอ้างอิง section
            # และตัวอย่างเนื้อหาจากเอกสารต้นฉบับ
            if contextualize:
                excerpt = path.read_text(encoding="utf-8-sig")[:6000]
                for chunk in file_chunks:
                    chunk.context = self.contextualize(chunk, excerpt)
            else:
                for chunk in file_chunks:
                    chunk.context = f"Section: {chunk.heading}; Source: {chunk.source}"
            chunks.extend(file_chunks)

        # ขั้นตอนที่ 4: สร้าง normalized embedding แบบ batch
        # เพื่อให้การคำนวณ cosine similarity มีสเกลสม่ำเสมอ
        texts = [self._embedding_text(chunk.retrieval_text) for chunk in chunks]
        vectors = self.embedder.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # ขั้นตอนที่ 5: เตรียมข้อมูลสำหรับ Vector index และ Lexical index
        vector_actions, lexical_actions = [], []
        for chunk, vector in zip(chunks, vectors):
            metadata = asdict(chunk)
            vector_actions.append(
                {
                    "_op_type": "index",
                    "_index": VECTOR_INDEX,
                    "_id": chunk.chunk_id,
                    "_source": {**metadata, "embedding": vector.tolist()},
                }
            )
            lexical_actions.append(
                {
                    "_op_type": "index",
                    "_index": LEXICAL_INDEX,
                    "_id": chunk.chunk_id,
                    "_source": metadata,
                }
            )
        # ขั้นตอนที่ 6: บันทึกข้อมูลแบบ bulk และ refresh เพื่อให้ค้นหาได้ทันที
        helpers.bulk(self.search_client, vector_actions)
        helpers.bulk(self.search_client, lexical_actions)
        self.search_client.indices.refresh(index=f"{VECTOR_INDEX},{LEXICAL_INDEX}")
        return len(chunks)

    def rebuild_lexical_from_vector(self) -> int:
        """สร้าง Lexical index ใหม่จาก chunks ใน Vector index."""
        if self.search_client.indices.exists(index=LEXICAL_INDEX):
            self.search_client.indices.delete(index=LEXICAL_INDEX)
        self.create_indices(recreate=False)
        actions = []
        for hit in helpers.scan(
            self.search_client,
            index=VECTOR_INDEX,
            query={"query": {"match_all": {}}, "_source": {"excludes": ["embedding"]}},
        ):
            actions.append(
                {
                    "_op_type": "index",
                    "_index": LEXICAL_INDEX,
                    "_id": hit["_id"],
                    "_source": hit["_source"],
                }
            )
        helpers.bulk(self.search_client, actions)
        self.search_client.indices.refresh(index=LEXICAL_INDEX)
        return len(actions)

    @staticmethod
    def _embedding_text(text: str) -> str:
        """เติม retrieval instruction ให้ passage ก่อนสร้าง embedding."""
        return f"Represent this passage for retrieval: {text}"

    def retrieve(self, question: str, top_k: int = 5, candidates: int = 24) -> list[dict]:
        """ค้นหา รวมคะแนน จัดอันดับใหม่ และเลือกหลักฐาน top-k."""

        # ขั้นตอนที่ 1: สร้าง normalized query embedding
        # Query ใช้ instruction คนละแบบกับ passage เพื่อระบุบทบาทของข้อความ
        query_vector = self.embedder.encode(
            f"Represent this question for retrieving relevant passages: {question}",
            normalize_embeddings=True,
        ).tolist()
        # ขั้นตอนที่ 2: Dense search ค้นจากความใกล้กันของความหมาย
        dense_response = self.search_client.search(
            index=VECTOR_INDEX,
            body={
                "size": candidates,
                "_source": {"excludes": ["embedding"]},
                "query": {"knn": {"embedding": {"vector": query_vector, "k": candidates}}},
            },
        )
        # ขั้นตอนที่ 3: Lexical search ค้นจากคำและ character n-gram ที่ตรงกัน
        lexical_response = self.search_client.search(
            index=LEXICAL_INDEX,
            body={
                "size": candidates,
                "query": {
                    "multi_match": {
                        "query": question,
                        "fields": ["heading^3", "content^2", "context^1.3"],
                        "type": "best_fields",
                    }
                },
            },
        )
        dense = dense_response["hits"]["hits"]
        lexical = lexical_response["hits"]["hits"]
        # ขั้นตอนที่ 4: รวม candidate จาก Dense และ Lexical ด้วย Weighted RRF
        combined = self._weighted_rrf(dense, lexical)

        # ขั้นตอนที่ 5: Cross-encoder reranking อ่านคำถามและ chunk พร้อมกัน
        # เพื่อวัดความสัมพันธ์โดยตรงอย่างละเอียดกว่า embedding similarity
        pairs = [(question, item["retrieval_text"]) for item in combined]
        rerank_raw = np.asarray(self.reranker.predict(pairs), dtype=float).reshape(-1)
        rerank_norm = self._minmax(rerank_raw)
        for item, score in zip(combined, rerank_norm):
            item["reranker_score"] = float(score)
            # ให้ reranker เป็นสัญญาณหลัก แต่ยังรักษาคะแนนจาก retrieval ไว้ 28%
            item["final_score"] = 0.72 * float(score) + 0.28 * item["retrieval_score"]

        # ขั้นตอนที่ 6: เรียงตามคะแนนรวมระหว่าง reranker และ retrieval
        combined.sort(key=lambda item: item["final_score"], reverse=True)

        # ขั้นตอนที่ 7: ลดผลลัพธ์ซ้ำและเลือกหลักฐานที่ครอบคลุมหลายประเด็น
        return self._diversify(combined, top_k)

    def _weighted_rrf(self, dense_hits: list[dict], lexical_hits: list[dict]) -> list[dict]:
        """รวมอันดับ Dense และ Lexical ด้วย Weighted RRF และ normalized scores.

        Dense มีน้ำหนัก 0.58 เพราะจับความหมายและคำพ้องได้ดี
        Lexical มีน้ำหนัก 0.42 เพื่อรักษาคำเฉพาะ ชื่อโรค และข้อความที่ตรงกัน
        จากนั้นผสม RRF 55%, normalized dense 30% และ normalized lexical 15%
        """
        records: dict[str, dict] = {}
        dense_norm = self._score_map(dense_hits)
        lexical_norm = self._score_map(lexical_hits)

        for channel, weight, hits in (("dense", 0.58, dense_hits), ("lexical", 0.42, lexical_hits)):
            for rank, hit in enumerate(hits, start=1):
                doc_id = hit["_id"]
                record = records.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        **hit["_source"],
                        "channels": [],
                        "rrf_score": 0.0,
                    },
                )
                record["channels"].append(channel)
                record["rrf_score"] += weight / (60 + rank)

        rrf_norm = self._minmax(np.array([item["rrf_score"] for item in records.values()]))
        for item, normalized_rrf in zip(records.values(), rrf_norm):
            d = dense_norm.get(item["doc_id"], 0.0)
            s = lexical_norm.get(item["doc_id"], 0.0)
            item["retrieval_score"] = 0.55 * float(normalized_rrf) + 0.30 * d + 0.15 * s
            item["retrieval_text"] = "\n".join(
                value for value in (item.get("heading", ""), item.get("context", ""), item.get("content", "")) if value
            )
        return sorted(records.values(), key=lambda item: item["retrieval_score"], reverse=True)

    def _diversify(self, ranked: list[dict], top_k: int, diversity: float = 0.15) -> list[dict]:
        """เลือก top-k โดยหักคะแนนเมื่อ chunk คล้ายผลที่เลือกไปแล้ว.

        แนวคิดคล้าย Maximal Marginal Relevance (MMR): รักษาความเกี่ยวข้อง
        ขณะเดียวกันเพิ่มความหลากหลายของหลักฐานที่ส่งให้ LLM
        """
        if not ranked:
            return []
        vectors = self.embedder.encode(
            [item["retrieval_text"] for item in ranked],
            normalize_embeddings=True,
        )
        selected: list[int] = []
        remaining = list(range(len(ranked)))
        while remaining and len(selected) < top_k:
            best_idx, best_value = None, -float("inf")
            for idx in remaining:
                redundancy = max((float(np.dot(vectors[idx], vectors[j])) for j in selected), default=0.0)
                value = ranked[idx]["final_score"] - diversity * redundancy
                if value > best_value:
                    best_idx, best_value = idx, value
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [ranked[idx] for idx in selected]

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        """ปรับคะแนนให้อยู่ในช่วง 0-1."""
        if values.size == 0:
            return values
        low, high = float(values.min()), float(values.max())
        if high - low < 1e-12:
            return np.ones_like(values, dtype=float)
        return (values - low) / (high - low)

    def _score_map(self, hits: list[dict]) -> dict[str, float]:
        """สร้างตาราง doc_id กับ normalized score."""
        values = np.array([float(hit["_score"]) for hit in hits])
        normalized = self._minmax(values)
        return {hit["_id"]: float(score) for hit, score in zip(hits, normalized)}

    def answer(self, question: str, top_k: int = 5) -> dict:
        """ค้นหลักฐานและให้ LLM เดิมสร้างคำตอบพร้อมข้อมูลอ้างอิง."""
        # ขั้นตอนที่ 1: เรียก retrieval pipeline เพื่อเลือกหลักฐานอันดับสูงสุด
        results = self.retrieve(question, top_k=top_k)

        # ขั้นตอนที่ 2: รวมเนื้อหา ชื่อไฟล์ ชื่อ section และ context
        # ให้อยู่ในรูปแบบที่ LLM อ่านและอ้างอิงได้ง่าย
        context = "\n\n".join(
            f"[Source: {item['source']} | Section: {item['heading']}]\n{item['content']}\nContext: {item['context']}"
            for item in results
        )
        # ขั้นตอนที่ 3: ส่งหลักฐานให้ LLM เดิมสร้างคำตอบสุดท้าย
        # v2 ไม่เปลี่ยนโมเดลในขั้นตอนนี้
        completion = self.llm.chat.completions.create(
            model=FINAL_ANSWER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Answer in Thai using only the supplied context. State clearly when the context is insufficient.",
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            max_tokens=1000,
            temperature=0.4,
        )
        # ขั้นตอนที่ 4: ตัดข้อมูลภายในที่ไม่จำเป็นออกก่อนส่งผลกลับ
        clean_results = [
            {key: value for key, value in item.items() if key != "retrieval_text"}
            for item in results
        ]
        return {
            "question": question,
            "answer": completion.choices[0].message.content,
            "results": clean_results,
            "models": {
                "embedding": DEFAULT_EMBEDDING_MODEL,
                "reranker": DEFAULT_RERANKER_MODEL,
                "final_answer": FINAL_ANSWER_MODEL,
            },
        }


def build_parser() -> argparse.ArgumentParser:
    """สร้าง CLI สำหรับ ingest, rebuild-lexical, search และ batch."""
    parser = argparse.ArgumentParser(description="Authentic RAG v2")
    parser.add_argument("--host", default=os.getenv("OPENSEARCH_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("OPENSEARCH_PORT", "9200")))
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Create v2 indices and ingest Markdown files")
    ingest.add_argument("--corpus", type=Path, required=True)
    ingest.add_argument("--recreate", action="store_true")
    ingest.add_argument("--no-contextualize", action="store_true")

    sub.add_parser("rebuild-lexical", help="Rebuild lexical index from existing v2 vector documents")

    search = sub.add_parser("search", help="Search and answer one question")
    search.add_argument("--question", required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--output", type=Path)

    batch = sub.add_parser("batch", help="Answer one question per line")
    batch.add_argument("--questions", type=Path, required=True)
    batch.add_argument("--top-k", type=int, default=5)
    batch.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """อ่านคำสั่ง CLI แล้วเรียก workflow ที่ผู้ใช้เลือก."""
    args = build_parser().parse_args()
    rag = AuthenticRAGv2(opensearch_host=args.host, opensearch_port=args.port)
    if args.command == "rebuild-lexical":
        count = rag.rebuild_lexical_from_vector()
        print(f"Rebuilt {LEXICAL_INDEX} with {count} chunks")
        return
    if args.command == "ingest":
        count = rag.ingest(
            args.corpus,
            recreate=args.recreate,
            contextualize=not args.no_contextualize,
        )
        print(f"Indexed {count} chunks into {VECTOR_INDEX} and {LEXICAL_INDEX}")
        return

    if args.command == "search":
        result = rag.answer(args.question, top_k=args.top_k)
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered)
        return

    questions = [line.strip() for line in args.questions.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    results = [rag.answer(question, top_k=args.top_k) for question in questions]
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
