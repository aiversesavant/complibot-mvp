import os
import re
import uuid
from typing import List, Dict, Tuple

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "complibot_documents"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

DOMAIN_TERMS = {
    "sop", "procedure", "process", "policy", "deviation", "deviations",
    "capa", "approval", "review", "training", "quality", "document",
    "documents", "triage", "adverse", "event", "verification", "audit",
    "compliance", "nonconformity", "non-conformity", "qa", "gxp"
}

INLINE_SECTION_LABELS = [
    "Objective", "Purpose", "Scope", "Policy", "Procedure",
    "Responsibilities", "Responsibility", "Identification", "Verification",
    "Approval", "Review", "Definitions", "Records", "References",
    "Corrective Action", "Preventive Action", "CAPA", "Deviation",
    "Training", "Drafting", "Initiation", "Investigation", "Closure",
    "Version Control", "Technical Review", "Quality Review", "Final Sign-off"
]


class CompliBotPipeline:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def reset_collection(self):
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def count_indexed_chunks(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)

    def clean_text(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" - ", "-")
        text = self._remove_filename_noise(text)
        return text.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        if not text:
            return []

        text = re.sub(r"\s+", " ", text).strip()
        text = self._remove_filename_noise(text)

        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = []

        for part in raw_sentences:
            part = self._normalize_sentence_text(part, ensure_period=False)
            part = self._strip_section_labels(part)
            part = self._remove_filename_noise(part)

            if self._is_sentence_fragment(part):
                continue

            if len(part.split()) >= 6:
                sentences.append(self._normalize_sentence_text(part, ensure_period=True))

        return sentences

    def split_into_sections(self, text: str) -> List[str]:
        if not text:
            return []

        text = self._remove_filename_noise(text)
        normalized = text.replace("\r", "\n").strip()

        labels_pattern = "|".join(re.escape(label) for label in INLINE_SECTION_LABELS)

        normalized = re.sub(
            rf"\s+(?=({labels_pattern})\s*:)",
            "\n\n",
            normalized,
            flags=re.IGNORECASE
        )

        parts = re.split(r"\n\s*\n", normalized)
        parts = [p.strip() for p in parts if len(p.strip()) > 30]

        if not parts:
            return [normalized]

        return parts

    def chunk_text(self, text: str, chunk_size: int = 700, overlap_sentences: int = 1) -> List[str]:
        sections = self.split_into_sections(text)
        if not sections:
            return []

        chunks = []

        for section in sections:
            section = self._remove_filename_noise(section)
            section = self._normalize_sentence_text(section, ensure_period=False)

            if len(section) <= chunk_size:
                if len(section.split()) >= 8:
                    chunks.append(section)
                continue

            sentences = self.split_into_sentences(section)
            if not sentences:
                continue

            current = []
            current_len = 0

            for sentence in sentences:
                s_len = len(sentence)
                if current_len + s_len + 1 <= chunk_size:
                    current.append(sentence)
                    current_len += s_len + 1
                else:
                    if current:
                        chunks.append(" ".join(current).strip())
                    overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
                    current = overlap + [sentence]
                    current_len = sum(len(x) + 1 for x in current)

            if current:
                chunks.append(" ".join(current).strip())

        cleaned_chunks = []
        for chunk in chunks:
            chunk = self._clean_snippet(chunk)
            chunk = self._remove_filename_noise(chunk)
            if len(chunk.split()) >= 8:
                cleaned_chunks.append(chunk)

        return cleaned_chunks

    def detect_doc_group(self, filename: str) -> str:
        f = filename.lower()

        if f.startswith("sop_") or "sop" in f:
            return "sop"

        if any(term in f for term in ["guideline", "gvp", "ich", "fda", "best-practices", "ema"]):
            return "guideline"

        if any(term in f for term in ["policy", "manual", "quality"]):
            return "quality_doc"

        return "general"

    def ingest_documents(self, uploaded_files) -> Tuple[int, List[str]]:
        self.reset_collection()
        os.makedirs("temp_uploads", exist_ok=True)

        total_chunks = 0
        processed_docs = []

        for uploaded_file in uploaded_files:
            temp_path = os.path.join("temp_uploads", uploaded_file.name)
            doc_group = self.detect_doc_group(uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            raw_text = self.extract_text_from_pdf(temp_path)
            cleaned = self.clean_text(raw_text)
            chunks = self.chunk_text(cleaned)

            if not chunks:
                continue

            embeddings = self.embedder.encode(chunks).tolist()
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [
                {
                    "source": uploaded_file.name,
                    "chunk_index": i,
                    "doc_group": doc_group
                }
                for i in range(len(chunks))
            ]

            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )

            processed_docs.append(uploaded_file.name)
            total_chunks += len(chunks)

        return total_chunks, processed_docs

    def classify_question(self, question: str) -> str:
        q = question.lower()

        if any(word in q for word in ["define", "what is", "meaning", "mean"]):
            return "Definition"
        if any(word in q for word in ["step", "steps", "process", "procedure", "how"]):
            return "Procedure"
        if any(word in q for word in ["policy", "requirement", "required", "must", "shall"]):
            return "Policy / Requirement"
        if any(word in q for word in ["training", "qualified", "qualification"]):
            return "Training"
        if any(word in q for word in ["escalation", "report", "notify", "deviation", "capa"]):
            return "Escalation / Quality Event"

        return "General Compliance Question"

    def infer_question_doc_preference(self, question: str) -> str:
        q = question.lower()

        if any(term in q for term in [
            "sop", "deviation", "capa", "approval process",
            "review process", "triage process", "document review"
        ]):
            return "sop"

        if any(term in q for term in [
            "guideline", "regulatory", "fda", "ich", "gcp", "gvp"
        ]):
            return "guideline"

        return "any"

    def retrieve_relevant_chunks(self, query: str, top_k: int = 4) -> List[Dict]:
        preferred_group = self.infer_question_doc_preference(query)

        primary_results = self._query_collection(
            query,
            top_k=top_k * 2,
            doc_group=preferred_group if preferred_group != "any" else None
        )

        if len(primary_results) < 2:
            primary_results = self._query_collection(query, top_k=top_k * 2, doc_group=None)

        reranked = self._rerank_results(query, primary_results)
        return reranked[:top_k]

    def _query_collection(self, query: str, top_k: int = 8, doc_group: str = None) -> List[Dict]:
        query_embedding = self.embedder.encode([query]).tolist()[0]

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }

        if doc_group:
            kwargs["where"] = {"doc_group": doc_group}

        results = self.collection.query(**kwargs)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved = []
        for doc, meta, dist in zip(docs, metas, distances):
            retrieved.append({
                "text": self._clean_snippet(doc),
                "source": meta.get("source", "Unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "doc_group": meta.get("doc_group", "general"),
                "distance": dist
            })

        return retrieved

    def _rerank_results(self, question: str, results: List[Dict]) -> List[Dict]:
        if not results:
            return []

        preferred_group = self.infer_question_doc_preference(question)

        source_counts = {}
        for r in results:
            source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1

        reranked = []
        for r in results:
            score = r["distance"]

            if preferred_group != "any" and r["doc_group"] == preferred_group:
                score -= 0.15

            if source_counts.get(r["source"], 0) >= 2:
                score -= 0.08

            if preferred_group == "sop" and r["doc_group"] == "guideline":
                score += 0.12

            reranked.append((score, r))

        reranked.sort(key=lambda x: x[0])
        return [r for _, r in reranked]

    def synthesize_answer(self, question: str, retrieved_chunks: List[Dict]) -> Dict:
        question_type = self.classify_question(question)

        if not retrieved_chunks:
            return {
                "question_type": question_type,
                "answer": "No relevant content was retrieved from the indexed compliance documents for this question.",
                "source": "No source found",
                "evidence": [],
                "compliance_note": (
                    "No document-grounded answer could be generated. "
                    "Please verify that relevant SOP or compliance content has been uploaded."
                )
            }

        grounding = self.evaluate_grounding(question, retrieved_chunks)

        if grounding["status"] == "not_grounded":
            return {
                "question_type": question_type,
                "answer": (
                    "This question does not appear to be sufficiently grounded in the uploaded compliance documents. "
                    "Please ask a document-based question about procedures, deviations, CAPA, approvals, training, "
                    "quality events, or related compliance content."
                ),
                "source": "No grounded source found",
                "evidence": [],
                "compliance_note": (
                    "The current retrieval results do not provide enough support for a reliable document-grounded answer."
                )
            }

        clustered_chunks = self._prefer_primary_source_cluster(retrieved_chunks)
        primary = clustered_chunks[0]
        answer = self._build_professional_answer(question, clustered_chunks, grounding["status"])
        evidence = self._build_evidence_snippets(question, clustered_chunks)

        compliance_note = (
            "This answer is based on the top retrieved content from the uploaded compliance documents "
            "and should be reviewed against approved internal procedures before operational use."
        )
        if grounding["status"] == "weakly_grounded":
            compliance_note = (
                "This answer is based on partially relevant retrieved content and should be reviewed carefully "
                "against approved internal procedures before operational use."
            )

        return {
            "question_type": question_type,
            "answer": answer,
            "source": f"{primary['source']} (chunk {primary['chunk_index']})",
            "evidence": evidence,
            "compliance_note": compliance_note
        }

    def _prefer_primary_source_cluster(self, retrieved_chunks: List[Dict]) -> List[Dict]:
        if not retrieved_chunks:
            return []

        source_counts = {}
        for r in retrieved_chunks:
            source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1

        primary_source = max(
            source_counts.items(),
            key=lambda x: (
                x[1],
                -min([c["distance"] for c in retrieved_chunks if c["source"] == x[0]])
            )
        )[0]

        primary_chunks = [r for r in retrieved_chunks if r["source"] == primary_source]
        other_chunks = [r for r in retrieved_chunks if r["source"] != primary_source]

        primary_chunks.sort(key=lambda x: x["distance"])
        other_chunks.sort(key=lambda x: x["distance"])

        return primary_chunks + other_chunks

    def evaluate_grounding(self, question: str, retrieved_chunks: List[Dict]) -> Dict:
        if not retrieved_chunks:
            return {"status": "not_grounded", "reason": "no_retrieval"}

        q_lower = question.lower().strip()

        obvious_chat_patterns = [
            "how are you", "who are you", "what is your name", "tell me a joke",
            "hello", "hi", "good morning", "good evening", "good afternoon", "what's up"
        ]
        if any(pattern in q_lower for pattern in obvious_chat_patterns):
            return {"status": "not_grounded", "reason": "casual_chat"}

        q_words = {
            w for w in re.findall(r"\w+", q_lower)
            if len(w) > 3 and w not in {"this", "that", "with", "from", "about", "does"}
        }
        q_domain_terms = {w for w in q_words if w in DOMAIN_TERMS}

        top_text = " ".join(chunk["text"].lower() for chunk in retrieved_chunks[:3])
        keyword_hits = sum(1 for word in q_words if word in top_text)
        domain_hits = sum(1 for word in q_domain_terms if word in top_text)

        best_distance = retrieved_chunks[0]["distance"]
        top_sources = [c["source"] for c in retrieved_chunks[:3]]
        same_source_count = max(top_sources.count(src) for src in set(top_sources)) if top_sources else 0

        if best_distance <= 0.95:
            return {"status": "strongly_grounded", "reason": "very_strong_top_match"}

        if best_distance <= 1.10 and (keyword_hits >= 1 or domain_hits >= 1):
            return {"status": "strongly_grounded", "reason": "strong_match_with_overlap"}

        if best_distance <= 1.25 and same_source_count >= 1 and len(q_domain_terms) >= 1:
            return {"status": "strongly_grounded", "reason": "strong_domain_top_match"}

        if best_distance <= 1.35 and domain_hits >= 1:
            return {"status": "strongly_grounded", "reason": "domain_overlap"}

        if best_distance <= 1.55 and (keyword_hits >= 1 or domain_hits >= 1):
            return {"status": "weakly_grounded", "reason": "acceptable_overlap"}

        if best_distance <= 1.60 and same_source_count >= 2:
            return {"status": "weakly_grounded", "reason": "same_source_weak"}

        return {"status": "not_grounded", "reason": "weak_match"}

    def _build_professional_answer(self, question: str, retrieved_chunks: List[Dict], grounding_status: str) -> str:
        top_text = " ".join(chunk["text"] for chunk in retrieved_chunks[:3])
        sentences = self.split_into_sentences(top_text)

        if not sentences:
            return "Relevant content was retrieved, but a clean answer could not be generated from the available text."

        qtype = self.classify_question(question)
        selected = self._select_relevant_sentences(question, sentences, max_sentences=3, question_type=qtype)
        if not selected:
            selected = sentences[:2]

        cleaned_sentences = []
        for sentence in selected:
            sentence = self._strip_section_labels(sentence)
            sentence = self._remove_filename_noise(sentence)
            sentence = self._normalize_sentence_text(sentence, ensure_period=True)
            if not self._is_sentence_fragment(sentence):
                cleaned_sentences.append(sentence)

        if not cleaned_sentences:
            cleaned_sentences = [
                self._normalize_sentence_text(
                    self._remove_filename_noise(self._strip_section_labels(s)),
                    ensure_period=True
                )
                for s in selected[:1]
            ]

        if qtype in {"Definition", "Procedure"}:
            cleaned_sentences = cleaned_sentences[:2]

        answer_body = " ".join(cleaned_sentences).strip()
        answer_body = self._trim_text(answer_body, 520)
        answer_body = self._lowercase_first(answer_body)

        q_lower = question.lower()

        if any(word in q_lower for word in ["what is", "define", "meaning", "mean"]):
            prefix = "The retrieved document content indicates that "
        elif any(word in q_lower for word in ["process", "procedure", "how", "steps"]):
            prefix = "Based on the retrieved SOP content, the process appears to be as follows: "
        elif any(word in q_lower for word in ["deviation", "capa", "review", "approval", "training"]):
            prefix = "The SOP indicates that "
        else:
            prefix = "Based on the retrieved compliance content, "

        if grounding_status == "weakly_grounded":
            prefix = "The retrieved content suggests that "

        return prefix + answer_body

    def _build_evidence_snippets(self, question: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        evidence = []
        seen_texts = set()
        qtype = self.classify_question(question)

        for item in retrieved_chunks[:3]:
            sentences = self.split_into_sentences(item["text"])
            selected = self._select_relevant_sentences(question, sentences, max_sentences=2, question_type=qtype)
            if not selected and sentences:
                selected = [sentences[0]]

            cleaned = []
            for sentence in selected:
                sentence = self._strip_section_labels(sentence)
                sentence = self._remove_filename_noise(sentence)
                sentence = self._normalize_sentence_text(sentence, ensure_period=True)
                if not self._is_sentence_fragment(sentence):
                    cleaned.append(sentence)

            if not cleaned:
                continue

            snippet = " ".join(cleaned).strip()
            snippet = self._trim_text(snippet, 280)

            if snippet not in seen_texts:
                seen_texts.add(snippet)
                evidence.append({
                    "source": item["source"],
                    "chunk_index": item["chunk_index"],
                    "distance": item["distance"],
                    "text": snippet
                })

        return evidence

    def _select_relevant_sentences(
        self,
        question: str,
        sentences: List[str],
        max_sentences: int = 2,
        question_type: str = "General Compliance Question"
    ) -> List[str]:
        q_words = {
            w for w in re.findall(r"\w+", question.lower())
            if len(w) > 3 and w not in {"does", "this", "that", "about", "from", "with", "into"}
        }

        scored = []
        for sentence in sentences:
            s_lower = sentence.lower()
            score = 0.0

            score += sum(1 for word in q_words if word in s_lower)

            if question_type in {"Definition", "Procedure"}:
                if any(label in s_lower for label in ["objective", "purpose", "scope"]):
                    score += 2.0
                if any(label in s_lower for label in ["process", "procedure", "steps", "review", "approval", "verification"]):
                    score += 1.5
            else:
                if any(label in s_lower for label in ["must", "shall", "required", "report", "approval", "review", "verification"]):
                    score += 1.5

            if len(sentence.split()) >= 8:
                score += 0.5

            scored.append((score, sentence))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        for score, sentence in scored:
            if score <= 0 and selected:
                continue
            selected.append(sentence)
            if len(selected) == max_sentences:
                break

        if question_type in {"Definition", "Procedure"} and len(selected) >= 2:
            selected = sorted(
                selected,
                key=lambda s: (
                    0 if any(k in s.lower() for k in ["objective", "purpose", "scope"]) else 1,
                    0 if len(s.split()) > 8 else 1
                )
            )

        return selected

    def _strip_section_labels(self, text: str) -> str:
        labels_pattern = "|".join(re.escape(label) for label in INLINE_SECTION_LABELS)
        text = re.sub(
            rf'^(?:{labels_pattern})\s*:\s*',
            '',
            text,
            flags=re.IGNORECASE
        )
        return text.strip()

    def _remove_filename_noise(self, text: str) -> str:
        text = re.sub(r'\bSOP_[A-Za-z0-9_]+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[A-Za-z0-9_]+\.pdf\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bSOP[A-Za-z0-9_\-]*\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def _normalize_sentence_text(self, text: str, ensure_period: bool = True) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace(" ,", ",").replace(" .", ".")
        text = text.replace(" :", ":").replace(" ;", ";")
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)

        if ensure_period and text and text[-1] not in ".!?":
            text += "."
        return text

    def _clean_snippet(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace(" - ", "-")
        text = self._remove_filename_noise(text)
        return text

    def _is_sentence_fragment(self, text: str) -> bool:
        words = text.split()
        if len(words) < 6:
            return True

        if words[-1].lower().strip(".") in {"the", "a", "an", "of", "to", "by", "for", "and", "or", "before"}:
            return True

        return False

    def _lowercase_first(self, text: str) -> str:
        if not text:
            return text
        return text[0].lower() + text[1:]

    def _trim_text(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        trimmed = text[:max_len].rsplit(" ", 1)[0]
        return trimmed + "..."