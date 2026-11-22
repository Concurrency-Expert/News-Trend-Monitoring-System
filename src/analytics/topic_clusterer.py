"""Incremental TF-IDF vectorization and cosine topic clustering for news articles."""

from __future__ import annotations

import copy
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from config.settings import settings
from src.analytics.text_preprocessor import IndonesianTextPreprocessor

_TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{2,}\b"
_TOKEN_EXTRACT_RE = re.compile(r"\b[a-zA-Z]{2,}\b")


def _parse_utc_datetime(value: Any, fallback: datetime) -> datetime:
    """Normalize datetime or ISO-8601 string into timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return fallback
    return fallback


def _extract_article_key(article: dict[str, Any]) -> str:
    """Return unique article key preferring `url_hash` with `_id` fallback."""
    url_hash = article.get("url_hash")
    if isinstance(url_hash, str) and url_hash.strip():
        return url_hash.strip()
    raw_id = article.get("_id")
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id).strip()
    raise ValueError("Article must define a non-empty 'url_hash' or '_id'.")


def _l2_normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalized float64 1D vector, leaving zero vectors unchanged."""
    arr = np.asarray(vec, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros_like(arr, dtype=np.float64)
    return arr / norm


@dataclass
class ClusteringResult:
    """Output bundle produced by a single batch run of TopicClusterer."""

    article_topic_assignments: dict[str, str]
    updated_topics: list[dict[str, Any]]
    new_topics: list[dict[str, Any]]
    unmatched_article_keys: list[str]
    vocabulary_state: dict[str, Any]


class IncrementalTfidfVectorizer:
    """Append-only TF-IDF vectorizer preserving feature indices across batches."""

    def __init__(self, max_features: int = 5000, sublinear_tf: bool = True) -> None:
        if max_features < 1:
            raise ValueError("max_features must be >= 1")
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.terms: list[str] = []
        self.term_to_idx: dict[str, int] = {}
        self.doc_freqs: list[int] = []
        self.total_docs: int = 0

    def export_state(self) -> dict[str, Any]:
        """Serialize vocabulary and document-frequency statistics."""
        return {
            "terms": list(self.terms),
            "doc_freqs": list(self.doc_freqs),
            "total_docs": int(self.total_docs),
            "max_features": int(self.max_features),
            "sublinear_tf": bool(self.sublinear_tf),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> IncrementalTfidfVectorizer:
        """Restore vectorizer from a previously exported state dictionary."""
        terms = list(state.get("terms", []))
        doc_freqs = [int(df) for df in state.get("doc_freqs", [])]
        if len(terms) != len(doc_freqs):
            raise ValueError("Invalid vocabulary state: terms and doc_freqs lengths differ.")

        instance = cls(
            max_features=int(state.get("max_features", 5000)),
            sublinear_tf=bool(state.get("sublinear_tf", True)),
        )
        instance.terms = terms
        instance.term_to_idx = {term: idx for idx, term in enumerate(terms)}
        instance.doc_freqs = doc_freqs
        instance.total_docs = int(state.get("total_docs", 0))
        return instance

    def _expand_vocabulary(self, doc_token_sets: list[set[str]]) -> None:
        """Append new batch terms deterministically up to `max_features`."""
        remaining = self.max_features - len(self.terms)
        if remaining <= 0:
            return

        batch_df: Counter[str] = Counter()
        for token_set in doc_token_sets:
            for tok in token_set:
                if tok not in self.term_to_idx:
                    batch_df[tok] += 1

        if not batch_df:
            return

        if len(batch_df) <= remaining:
            new_terms = sorted(batch_df.keys())
        else:
            # Keep the most common batch terms when hitting max_features capacity.
            new_terms = sorted(batch_df.keys(), key=lambda t: (-batch_df[t], t))[:remaining]

        for term in new_terms:
            idx = len(self.terms)
            self.terms.append(term)
            self.term_to_idx[term] = idx
            self.doc_freqs.append(0)

    def fit_transform_batch(self, processed_docs: Sequence[str]) -> np.ndarray:
        """Update global DF/IDF statistics with a new batch and return L2 TF-IDF vectors."""
        docs = [doc if isinstance(doc, str) else "" for doc in processed_docs]
        if not docs:
            return np.empty((0, len(self.terms)), dtype=np.float64)

        doc_token_sets = [set(_TOKEN_EXTRACT_RE.findall(doc)) for doc in docs]
        self._expand_vocabulary(doc_token_sets)

        self.total_docs += len(docs)
        for token_set in doc_token_sets:
            for tok in token_set:
                idx = self.term_to_idx.get(tok)
                if idx is not None:
                    self.doc_freqs[idx] += 1

        if not self.terms:
            return np.zeros((len(docs), 0), dtype=np.float64)

        cv = CountVectorizer(
            lowercase=False,
            token_pattern=_TOKEN_PATTERN,
            vocabulary=self.term_to_idx,
            dtype=np.float64,
        )
        tf_matrix = cv.transform(docs).toarray()

        if self.sublinear_tf:
            nonzero_mask = tf_matrix > 0.0
            tf_matrix[nonzero_mask] = 1.0 + np.log(tf_matrix[nonzero_mask])

        df_array = np.asarray(self.doc_freqs, dtype=np.float64)
        idf = np.log((1.0 + self.total_docs) / (1.0 + df_array)) + 1.0
        tfidf_matrix = tf_matrix * idf

        return normalize(tfidf_matrix, norm="l2", axis=1, copy=False)

    def align_vector(self, vector: Sequence[float]) -> np.ndarray:
        """Zero-pad or trim a stored centroid vector to match current vocabulary width."""
        width = len(self.terms)
        if width == 0:
            return np.zeros(0, dtype=np.float64)

        arr = np.asarray(vector, dtype=np.float64).ravel()
        if arr.size == width:
            return _l2_normalize_vector(arr)
        if arr.size < width:
            padded = np.zeros(width, dtype=np.float64)
            padded[: arr.size] = arr
            return _l2_normalize_vector(padded)
        return _l2_normalize_vector(arr[:width])


class TopicClusterer:
    """Phase 2 NLP engine: preprocessing, centroid matching, discovery, auto-labeling."""

    def __init__(
        self,
        similarity_threshold: float = settings.NLP_SIMILARITY_THRESHOLD,
        min_cluster_size: int = settings.NLP_MIN_CLUSTER_SIZE,
        top_keywords: int = settings.NLP_TOP_KEYWORDS,
        label_keywords: int = settings.NLP_LABEL_KEYWORDS,
        max_features: int = settings.NLP_MAX_FEATURES,
        title_weight: int = settings.NLP_TITLE_WEIGHT,
        use_stemming: bool = settings.NLP_USE_STEMMING,
        preprocessor: Optional[IndonesianTextPreprocessor] = None,
        vectorizer: Optional[IncrementalTfidfVectorizer] = None,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0.0, 1.0]")
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2")
        if top_keywords < 1 or label_keywords < 1:
            raise ValueError("top_keywords and label_keywords must be >= 1")

        self.similarity_threshold = float(similarity_threshold)
        self.min_cluster_size = int(min_cluster_size)
        self.top_keywords = int(top_keywords)
        self.label_keywords = int(label_keywords)
        self.title_weight = int(title_weight)

        self.preprocessor = preprocessor or IndonesianTextPreprocessor(
            use_stemming=use_stemming
        )
        self.vectorizer = vectorizer or IncrementalTfidfVectorizer(
            max_features=max_features
        )

    def extract_keywords_and_label(
        self,
        centroid: Sequence[float] | np.ndarray,
        surface_counts: Optional[dict[str, Counter[str]]] = None,
    ) -> tuple[list[str], str]:
        """Extract deterministic top-K keywords and a Title-Case label from centroid."""
        vec = np.asarray(centroid, dtype=np.float64).ravel()
        if vec.size == 0 or not self.vectorizer.terms:
            return [], ""

        positive_indices = [idx for idx, w in enumerate(vec) if w > 0.0 and np.isfinite(w)]
        if not positive_indices:
            return [], ""

        ranked_indices = sorted(
            positive_indices,
            key=lambda idx: (-float(vec[idx]), self.vectorizer.terms[idx]),
        )

        keywords: list[str] = []
        seen: set[str] = set()
        for idx in ranked_indices:
            term = self.vectorizer.terms[idx]
            chosen = term
            if surface_counts and term in surface_counts and surface_counts[term]:
                # Deterministic tie-break: highest surface frequency, then alphabetical.
                chosen = sorted(
                    surface_counts[term].items(),
                    key=lambda item: (-item[1], item[0]),
                )[0][0]
            if chosen not in seen:
                seen.add(chosen)
                keywords.append(chosen)
            if len(keywords) >= self.top_keywords:
                break

        label_tokens = [kw.title() for kw in keywords[: self.label_keywords]]
        return keywords, " ".join(label_tokens)

    @staticmethod
    def _next_topic_seq(topics: Sequence[dict[str, Any]], date_str: str) -> int:
        """Compute next non-colliding integer sequence for `topik_{YYYYMMDD}_{seq:03d}`."""
        pattern = re.compile(rf"^topik_{re.escape(date_str)}_(\d+)$")
        max_seq = 0
        for topic in topics:
            topic_id = str(topic.get("topic_id", ""))
            match = pattern.match(topic_id)
            if match:
                max_seq = max(max_seq, int(match.group(1)))
        return max_seq + 1

    @staticmethod
    def _build_surface_map(pairs: Sequence[tuple[str, str]]) -> dict[str, Counter[str]]:
        """Accumulate surface-token frequencies per normalized stem."""
        surface_map: dict[str, Counter[str]] = {}
        for term, surface in pairs:
            counter = surface_map.get(term)
            if counter is None:
                counter = Counter()
                surface_map[term] = counter
            counter[surface] += 1
        return surface_map

    @staticmethod
    def _merge_surface_maps(
        maps: Sequence[dict[str, Counter[str]]],
    ) -> dict[str, Counter[str]]:
        merged: dict[str, Counter[str]] = {}
        for m in maps:
            for term, counter in m.items():
                if term not in merged:
                    merged[term] = Counter(counter)
                else:
                    merged[term].update(counter)
        return merged

    def cluster_articles(
        self,
        articles: Sequence[dict[str, Any]],
        active_topics: Optional[Sequence[dict[str, Any]]] = None,
        batch_time: Optional[datetime] = None,
    ) -> ClusteringResult:
        """Cluster unassigned articles against active topics and discover new clusters."""
        now_utc = _parse_utc_datetime(
            batch_time, datetime.now(timezone.utc)
        )
        all_existing_topics = list(active_topics or [])

        # Filter to unassigned articles (`topic_id is None`) and sort deterministically.
        pending: list[tuple[datetime, str, dict[str, Any]]] = []
        for article in articles:
            if article.get("topic_id") is not None:
                continue
            key = _extract_article_key(article)
            pub_dt = _parse_utc_datetime(
                article.get("published_at") or article.get("created_at"),
                now_utc,
            )
            pending.append((pub_dt, key, article))

        pending.sort(key=lambda item: (item[0], item[1]))

        if not pending:
            return ClusteringResult(
                article_topic_assignments={},
                updated_topics=[],
                new_topics=[],
                unmatched_article_keys=[],
                vocabulary_state=self.vectorizer.export_state(),
            )

        # Preprocess and vectorize the incoming batch together.
        processed_texts: list[str] = []
        surface_maps: list[dict[str, Counter[str]]] = []
        for _, _, article in pending:
            proc_text, pairs = self.preprocessor.preprocess_article(
                article.get("title"),
                article.get("clean_text"),
                title_weight=self.title_weight,
            )
            processed_texts.append(proc_text)
            surface_maps.append(self._build_surface_map(pairs))

        vectors = self.vectorizer.fit_transform_batch(processed_texts)

        # Prepare deep copies of active topics aligned to current vocabulary width.
        active_states: list[dict[str, Any]] = []
        for topic in all_existing_topics:
            if topic.get("status", "active") != "active":
                continue
            topic_copy = copy.deepcopy(topic)
            aligned = self.vectorizer.align_vector(topic_copy.get("centroid_vector", []))
            topic_copy["_centroid_np"] = aligned
            topic_copy["_dirty"] = False
            topic_copy["_surface_maps"] = []
            topic_copy["last_updated_at"] = _parse_utc_datetime(
                topic_copy.get("last_updated_at"), now_utc
            )
            if topic_copy.get("first_seen_at") is not None:
                topic_copy["first_seen_at"] = _parse_utc_datetime(
                    topic_copy.get("first_seen_at"), now_utc
                )
            active_states.append(topic_copy)

        active_states.sort(key=lambda t: str(t.get("topic_id", "")))

        assignments: dict[str, str] = {}
        unmatched_candidates: list[int] = []
        unmatched_article_keys: list[str] = []

        # Step C: Centroid matching against active topics.
        for idx, (pub_dt, key, _) in enumerate(pending):
            vec = vectors[idx]
            if float(np.linalg.norm(vec)) <= 0.0:
                unmatched_article_keys.append(key)
                continue

            if active_states:
                centroid_matrix = np.vstack([t["_centroid_np"] for t in active_states])
                sims = cosine_similarity(vec.reshape(1, -1), centroid_matrix)[0]
                best_pos = int(np.argmax(sims))
                best_sim = float(sims[best_pos])
            else:
                best_pos = -1
                best_sim = -1.0

            if best_pos >= 0 and best_sim >= self.similarity_threshold:
                matched = active_states[best_pos]
                assignments[key] = str(matched["topic_id"])

                prev_count = max(int(matched.get("article_count", 0)), 0)
                updated_centroid = _l2_normalize_vector(
                    (prev_count * matched["_centroid_np"] + vec) / (prev_count + 1)
                )
                matched["_centroid_np"] = updated_centroid
                matched["article_count"] = prev_count + 1
                matched["last_updated_at"] = max(matched["last_updated_at"], pub_dt)
                matched["_dirty"] = True
                matched["_surface_maps"].append(surface_maps[idx])
            else:
                unmatched_candidates.append(idx)

        updated_topics: list[dict[str, Any]] = []
        for topic_state in active_states:
            if not topic_state["_dirty"]:
                continue
            merged_surfaces = self._merge_surface_maps(topic_state["_surface_maps"])
            # Preserve existing keyword spelling preference if already defined.
            for existing_kw in topic_state.get("keywords", []):
                if isinstance(existing_kw, str) and existing_kw.strip():
                    stem_pairs = self.preprocessor.tokenize_with_surfaces(existing_kw)
                    for term, surf in stem_pairs:
                        merged_surfaces.setdefault(term, Counter())[surf] += 1

            keywords, generated_label = self.extract_keywords_and_label(
                topic_state["_centroid_np"], merged_surfaces
            )
            centroid_list = [float(v) for v in topic_state["_centroid_np"].tolist()]

            cleaned_topic = {
                k: v for k, v in topic_state.items() if not k.startswith("_")
            }
            cleaned_topic["centroid_vector"] = centroid_list
            cleaned_topic["keywords"] = keywords
            if not cleaned_topic.get("label"):
                cleaned_topic["label"] = generated_label
            updated_topics.append(cleaned_topic)

        # Step D: Dynamic discovery on remaining unmatched articles.
        new_topics: list[dict[str, Any]] = []
        if len(unmatched_candidates) < max(2, self.min_cluster_size):
            for idx in unmatched_candidates:
                unmatched_article_keys.append(pending[idx][1])
        else:
            sub_vectors = vectors[unmatched_candidates]
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=1.0 - self.similarity_threshold,
            )
            labels = clustering.fit_predict(sub_vectors)

            groups: dict[int, list[int]] = {}
            for local_i, cluster_label in enumerate(labels):
                groups.setdefault(int(cluster_label), []).append(
                    unmatched_candidates[local_i]
                )

            # Sort discovered candidate clusters deterministically by earliest article & key.
            ordered_groups = sorted(
                groups.values(),
                key=lambda member_idxs: (
                    min(pending[i][0] for i in member_idxs),
                    min(pending[i][1] for i in member_idxs),
                ),
            )

            date_str = now_utc.strftime("%Y%m%d")
            next_seq = self._next_topic_seq(all_existing_topics, date_str)

            for member_idxs in ordered_groups:
                if len(member_idxs) < self.min_cluster_size:
                    for idx in member_idxs:
                        unmatched_article_keys.append(pending[idx][1])
                    continue

                topic_id = f"topik_{date_str}_{next_seq:03d}"
                next_seq += 1

                centroid_vec = _l2_normalize_vector(
                    np.mean(vectors[member_idxs], axis=0)
                )
                merged_surfaces = self._merge_surface_maps(
                    [surface_maps[i] for i in member_idxs]
                )
                keywords, label = self.extract_keywords_and_label(
                    centroid_vec, merged_surfaces
                )

                timestamps = [pending[i][0] for i in member_idxs]
                for idx in member_idxs:
                    assignments[pending[idx][1]] = topic_id

                new_topics.append(
                    {
                        "topic_id": topic_id,
                        "label": label,
                        "keywords": keywords,
                        "centroid_vector": [float(v) for v in centroid_vec.tolist()],
                        "status": "active",
                        "article_count": int(len(member_idxs)),
                        "first_seen_at": min(timestamps),
                        "last_updated_at": max(timestamps),
                    }
                )

        return ClusteringResult(
            article_topic_assignments=assignments,
            updated_topics=updated_topics,
            new_topics=new_topics,
            unmatched_article_keys=unmatched_article_keys,
            vocabulary_state=self.vectorizer.export_state(),
        )
