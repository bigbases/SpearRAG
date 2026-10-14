"""SpearRAG retriever: Subgraph Propagation with Entity-bridged Adaptive Reranking (retrieval part).

LLM-free indexing and retrieval. The learned reranker (stage 3) lives in ``reranker.py``.

Offline -- Entity-Shared Passage Graph Construction (``index``)
    spaCy NER over every passage; passage / entity / sentence embeddings; passage x entity TF-IDF
    incidence; a top-k passage graph whose edges are the cosine of entity TF-IDF vectors
    (entities shared by two passages), built block-wise so memory stays linear in corpus size.

Online, per query (``retrieve``)
    1. Entity Bridging (E)
       - Entity Seeding           question NER mentions -> indexed entities (``_link_query_entities``)
       - Bridge Propagation       up to 3 hops along the query-most-similar sentences (``_bridge_propagation``)
       - Bridged Candidate Gen.   dense top-128 U bridge passages U graph neighbours (``_collect_candidates``)
    2. Subgraph Propagation (SP)
       - Query-Induced Subgraph   candidates + their entities, entity-containment and passage-graph edges
       - Subgraph PPR             restart = active entities (75%) + dense top-96 passages (25%), alpha=0.85
                                  (``_subgraph_ppr``)
    3. Adaptive Reranking (AR)
       - Score Fusion             PPR / dense / entity coverage (``_fuse_scores``) -- the reranker's input
       - Graph-Aware Listwise Reranker   see ``reranker.py``

``retrieve`` returns the fused ranking (no learned reranker); ``reranker.AdaptiveReranker`` is the full method.

Provenance: this is the community-free configuration of the research code's ``QACFRetriever``
(``final_test_eval.py --no-community``) with the unused community machinery removed. For every query the
candidate set, PPR, coverage and fusion scores are identical to that code (checked by
``scripts/check_equivalence.py``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from tqdm import tqdm

from spearrag.config import ProjectConfig
from spearrag.embedding_store import EmbeddingStore
from spearrag.ner import SpacyNER
from spearrag.utils import compute_mdhash_id

logger = logging.getLogger(__name__)


@dataclass
class SpearRAGConfig:
    """Retrieval hyper-parameters (paper defaults)."""

    retrieval_top_k: int = 5

    # Offline: entity-shared passage graph
    passage_graph_top_k: int = 24                 # neighbours kept per passage
    passage_graph_min_weight: float = 0.05
    max_entity_document_frequency_ratio: float = 0.20   # entities in > 20% of passages are ignored ...
    max_entity_document_frequency: int = 5000     # ... and in > 5000 passages (keeps graph size linear)
    passage_graph_block_size: int = 2048          # rows per block of the similarity product

    # (1) Entity Bridging -- seeding
    seed_candidates_per_mention: int = 2
    seed_similarity_threshold: float = 0.55
    seed_similarity_margin: float = 0.06
    # (1) Entity Bridging -- bridge propagation
    max_bridge_hops: int = 3                      # 0 = seeds only (ablation "no_bridge")
    sentences_per_entity: int = 3
    sentence_similarity_floor: float = 0.18
    activation_threshold: float = 0.06
    hop_decay: float = 0.90
    max_active_entities: int = 256
    # (1) Entity Bridging -- bridged candidate generation
    enable_bridge_candidates: bool = True         # passages holding activated entities
    global_dense_candidates: int = 128            # dense top-N over the whole corpus
    max_bridge_passages: int = 512
    passage_neighbor_expansion: int = 2           # passage-graph neighbours added per candidate
    max_candidate_passages: int = 4096

    # (2) Subgraph Propagation
    enable_subgraph_ppr: bool = True              # False = restart distribution only (ablation)
    ppr_scope: str = "subgraph"                   # "global": same restart on the full graph (control)
    passage_graph_edge_weight: float = 0.15
    ppr_damping: float = 0.85
    ppr_tolerance: float = 1e-8
    ppr_max_iterations: int = 80
    restart_entity_weight: float = 0.75
    dense_restart_top_n: int = 96
    dense_temperature: float = 0.08

    # (3) Score fusion (normalised automatically)
    fusion_ppr_weight: float = 0.60
    fusion_dense_weight: float = 0.18
    fusion_coverage_weight: float = 0.08

    # Evidence order handed to the generator when the fused ranking is used directly
    enable_chain_ordering: bool = True
    generation_pool_size: int = 20
    generation_connectivity_weight: float = 0.30

    def validate(self) -> None:
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive")
        if self.passage_graph_top_k < 1:
            raise ValueError("passage_graph_top_k must be positive")
        if not 0.0 <= self.ppr_damping < 1.0:
            raise ValueError("ppr_damping must be in [0, 1)")
        if self.ppr_scope not in ("subgraph", "global"):
            raise ValueError("ppr_scope must be 'subgraph' or 'global'")


# ----------------------------------------------------------------------------- numeric helpers

def _l1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x) & (x > 0.0), x, 0.0)
    total = float(x.sum())
    return x / total if total > 0.0 else np.zeros_like(x)


def _row_normalize_dense(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _softmax_sparse_values(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    z = (values - values.max()) / max(float(temperature), 1e-6)
    z = np.exp(np.clip(z, -50.0, 0.0))
    return _l1(z)


def _topk_rows(matrix: csr_matrix, k: int, min_value: float = 0.0) -> csr_matrix:
    """Keep at most k largest positive entries per sparse row."""
    matrix = matrix.tocsr()
    out_indices: list[int] = []
    out_values: list[float] = []
    out_indptr = [0]
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        idx = matrix.indices[start:end]
        val = matrix.data[start:end]
        keep_mask = np.isfinite(val) & (val > min_value)
        idx, val = idx[keep_mask], val[keep_mask]
        if val.size > k:
            chosen = np.argpartition(val, -k)[-k:]
            idx, val = idx[chosen], val[chosen]
        if val.size:
            order = np.argsort(val)[::-1]
            out_indices.extend(idx[order].tolist())
            out_values.extend(val[order].astype(float).tolist())
        out_indptr.append(len(out_indices))
    return csr_matrix(
        (
            np.asarray(out_values, dtype=np.float32),
            np.asarray(out_indices, dtype=np.int32),
            np.asarray(out_indptr, dtype=np.int64),
        ),
        shape=matrix.shape,
    )


def _row_stochastic(adjacency: csr_matrix) -> tuple[csr_matrix, np.ndarray]:
    adjacency = adjacency.tocsr().astype(np.float64)
    adjacency.data = np.where(np.isfinite(adjacency.data) & (adjacency.data > 0.0), adjacency.data, 0.0)
    adjacency.eliminate_zeros()
    row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sum)
    nonzero = row_sum > 0.0
    inv[nonzero] = 1.0 / row_sum[nonzero]
    return (sp.diags(inv) @ adjacency).tocsr(), ~nonzero


def sparse_ppr(adjacency: csr_matrix, restart: np.ndarray, damping: float,
               tolerance: float, max_iterations: int) -> np.ndarray:
    """Personalized PageRank with restart-aware dangling-node handling."""
    restart = _l1(restart)
    if restart.sum() == 0.0:
        restart = np.full(adjacency.shape[0], 1.0 / max(adjacency.shape[0], 1))
    transition, dangling = _row_stochastic(adjacency)
    score = restart.copy()
    for _ in range(max_iterations):
        dangling_mass = float(score[dangling].sum()) if dangling.any() else 0.0
        updated = (
            (1.0 - damping) * restart
            + damping * np.asarray(transition.T @ score).ravel()
            + damping * dangling_mass * restart
        )
        if np.abs(updated - score).sum() <= tolerance:
            score = updated
            break
        score = updated
    return _l1(score)


def _ordered_store_hids(store: EmbeddingStore) -> list[str]:
    """Hash IDs in exactly the order of ``store.embeddings``."""
    return [hid for hid, _ in sorted(store.hash_id_to_idx.items(), key=lambda item: item[1])]


# ----------------------------------------------------------------------------- retriever

class SpearRAGRetriever:
    """Indexing and stages 1-2 (+ score fusion) of SpearRAG."""

    def __init__(self, project_config: ProjectConfig, config: Optional[SpearRAGConfig] = None) -> None:
        self.project = project_config
        self.cfg = config or SpearRAGConfig()
        self.cfg.validate()
        self.base = os.path.join(project_config.working_dir, project_config.dataset_name, "spearrag")
        os.makedirs(self.base, exist_ok=True)

        # Stores are corpus-fingerprint-specific and initialised by index().
        self.passage_store: Optional[EmbeddingStore] = None
        self.entity_store: Optional[EmbeddingStore] = None
        self.sentence_store: Optional[EmbeddingStore] = None
        self.ner = SpacyNER(project_config.spacy_model)

        self._indexed = False
        self._passage_hids: list[str] = []
        self._entity_hids: list[str] = []
        self._sentence_hids: list[str] = []
        self._passage_texts: list[str] = []
        self._entity_texts: list[str] = []
        self._sentence_texts: list[str] = []
        self._passage_embs = np.empty((0, 0), dtype=np.float32)
        self._entity_embs = np.empty((0, 0), dtype=np.float32)
        self._sentence_embs = np.empty((0, 0), dtype=np.float32)

        self._contain = csr_matrix((0, 0), dtype=np.float32)       # passage x entity TF-IDF
        self._contain_bin = csr_matrix((0, 0), dtype=np.float32)   # passage x entity binary
        self._mention = csr_matrix((0, 0), dtype=np.float32)       # sentence x entity binary
        self._passage_graph = csr_matrix((0, 0), dtype=np.float32)
        self._entity_to_sentences: list[np.ndarray] = []
        self._sentence_to_entities: list[np.ndarray] = []
        self._entity_casefold_to_indices: dict[str, list[int]] = defaultdict(list)
        self._global_adjacency: Optional[csr_matrix] = None

    # ------------------------------------------------------------------ offline: indexing

    def index(self, passages: list[str]) -> None:
        """Entity-Shared Passage Graph Construction."""
        if not passages:
            raise ValueError("passages must not be empty")
        fingerprint = self._corpus_fingerprint(passages)
        ner_path = os.path.join(self.base, f"ner_{fingerprint}.json")
        self._initialize_embedding_stores(fingerprint)
        hash_id_to_passage = {compute_mdhash_id(text, prefix="passage-"): text for text in passages}
        if len(hash_id_to_passage) != len(passages):
            logger.warning("Duplicate passage texts were collapsed by content hash")

        # With a transformer spaCy model on the GPU, NER and passage encoding both want the GPU and
        # cupy's device is thread-local, so run them sequentially; on CPU they overlap.
        from spearrag.ner import _SPACY_ON_GPU
        if _SPACY_ON_GPU:
            passage_to_entities, sentence_to_entities = self._load_or_run_ner(hash_id_to_passage, ner_path)
            self.passage_store.insert_text(list(hash_id_to_passage.values()))
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                ner_future = executor.submit(self._load_or_run_ner, hash_id_to_passage, ner_path)
                self.passage_store.insert_text(list(hash_id_to_passage.values()))
                passage_to_entities, sentence_to_entities = ner_future.result()

        entity_texts = sorted({e for entities in passage_to_entities.values() for e in entities})
        sentence_texts = sorted(sentence_to_entities)
        self.entity_store.insert_text(entity_texts)
        self.sentence_store.insert_text(sentence_texts)

        self._load_store_arrays()
        self._build_incidence_matrices(passage_to_entities, sentence_to_entities)
        self._build_passage_graph()
        self._build_bridge_lookups()
        self._global_adjacency = None
        self._indexed = True
        logger.info("SpearRAG index complete: %d passages, %d entities, %d sentences",
                    len(self._passage_hids), len(self._entity_hids), len(self._sentence_hids))

    def _corpus_fingerprint(self, passages: Iterable[str]) -> str:
        digest = hashlib.sha256()
        model = self.project.embedding_model
        model_identity = getattr(model, "model_name_or_path",
                                 getattr(model, "name_or_path", type(model).__qualname__))
        digest.update(str(model_identity).encode("utf-8"))
        digest.update(str(self.project.spacy_model).encode("utf-8"))
        for text in passages:
            digest.update(text.encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        return digest.hexdigest()[:16]

    def _initialize_embedding_stores(self, fingerprint: str) -> None:
        model, batch_size = self.project.embedding_model, self.project.batch_size
        self.passage_store = EmbeddingStore(
            model, os.path.join(self.base, f"passage_embedding_{fingerprint}.parquet"), batch_size, "passage")
        self.entity_store = EmbeddingStore(
            model, os.path.join(self.base, f"entity_embedding_{fingerprint}.parquet"), batch_size, "entity")
        self.sentence_store = EmbeddingStore(
            model, os.path.join(self.base, f"sentence_embedding_{fingerprint}.parquet"), batch_size, "sentence")

    def _load_or_run_ner(self, hash_id_to_passage: dict[str, str], cache_path: str
                         ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            return cached["passage_hash_id_to_entities"], cached["sentence_to_entities"]
        passage_to_entities, sentence_to_entities = self.ner.batch_ner(
            hash_id_to_passage, self.project.max_workers, self.project.ner_n_process)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump({"passage_hash_id_to_entities": passage_to_entities,
                       "sentence_to_entities": sentence_to_entities}, handle, ensure_ascii=False)
        return passage_to_entities, sentence_to_entities

    def _load_store_arrays(self) -> None:
        self._passage_hids = _ordered_store_hids(self.passage_store)
        self._entity_hids = _ordered_store_hids(self.entity_store)
        self._sentence_hids = _ordered_store_hids(self.sentence_store)
        self._passage_texts = [self.passage_store.hash_id_to_text[h] for h in self._passage_hids]
        self._entity_texts = [self.entity_store.hash_id_to_text[h] for h in self._entity_hids]
        self._sentence_texts = [self.sentence_store.hash_id_to_text[h] for h in self._sentence_hids]
        self._passage_embs = _row_normalize_dense(np.asarray(self.passage_store.embeddings, dtype=np.float32))
        self._entity_embs = _row_normalize_dense(np.asarray(self.entity_store.embeddings, dtype=np.float32))
        self._sentence_embs = _row_normalize_dense(np.asarray(self.sentence_store.embeddings, dtype=np.float32))
        self._entity_casefold_to_indices.clear()
        for index, text in enumerate(self._entity_texts):
            self._entity_casefold_to_indices[text.casefold()].append(index)

    def _build_incidence_matrices(self, passage_to_entities: dict[str, list[str]],
                                  sentence_to_entities: dict[str, list[str]]) -> None:
        p_index = {hid: i for i, hid in enumerate(self._passage_hids)}
        e_index = {text: i for i, text in enumerate(self._entity_texts)}
        s_index = {text: i for i, text in enumerate(self._sentence_texts)}

        # Mention counts come from the NER output (no substring matching).
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        for passage_hid, entities in passage_to_entities.items():
            if passage_hid not in p_index:
                continue
            pi = p_index[passage_hid]
            for entity in entities:
                if entity in e_index:
                    pair_counts[(pi, e_index[entity])] += 1
        rows = np.fromiter((key[0] for key in pair_counts), dtype=np.int32)
        cols = np.fromiter((key[1] for key in pair_counts), dtype=np.int32)
        counts = np.fromiter(pair_counts.values(), dtype=np.float32)
        count_matrix = csr_matrix((counts, (rows, cols)), shape=(len(self._passage_hids), len(self._entity_hids)))
        self._contain_bin = count_matrix.copy()
        self._contain_bin.data[:] = 1.0

        document_frequency = np.asarray(self._contain_bin.sum(axis=0)).ravel()
        n_passages = max(len(self._passage_hids), 1)
        idf = np.log((n_passages + 1.0) / (document_frequency + 1.0)) + 1.0
        tf = count_matrix.copy()
        tf.data = 1.0 + np.log(tf.data)
        self._contain = (tf @ sp.diags(idf.astype(np.float32))).tocsr()

        mention_rows: list[int] = []
        mention_cols: list[int] = []
        for sentence, entities in sentence_to_entities.items():
            si = s_index.get(sentence)
            if si is None:
                continue
            for entity in set(entities):
                ei = e_index.get(entity)
                if ei is not None:
                    mention_rows.append(si)
                    mention_cols.append(ei)
        self._mention = csr_matrix((np.ones(len(mention_rows), dtype=np.float32), (mention_rows, mention_cols)),
                                   shape=(len(self._sentence_hids), len(self._entity_hids)))

    def _build_passage_graph(self) -> None:
        """Top-k passage graph from shared entities (cosine of entity TF-IDF vectors)."""
        contain = self._contain.copy().tocsr()
        # Very common entities create quadratic cliques and semantic noise; drop them for the graph only.
        df = np.asarray(self._contain_bin.sum(axis=0)).ravel()
        df_limit = max(2, min(int(len(self._passage_hids) * self.cfg.max_entity_document_frequency_ratio),
                              self.cfg.max_entity_document_frequency))
        contain = contain[:, df <= df_limit]

        row_norm = np.sqrt(np.asarray(contain.multiply(contain).sum(axis=1)).ravel())
        inv = np.zeros_like(row_norm)
        inv[row_norm > 0.0] = 1.0 / row_norm[row_norm > 0.0]
        normalized = (sp.diags(inv) @ contain).tocsr()
        normalized_t = normalized.T.tocsc()

        # Block-wise product: the N x N similarity matrix is never materialised.
        n_rows = normalized.shape[0]
        block = max(1, self.cfg.passage_graph_block_size)
        parts = []
        for start in range(0, n_rows, block):
            blk = (normalized[start:start + block] @ normalized_t).tocsr()
            blk.setdiag(0.0, k=start)        # drop self-similarity (row i <-> passage start+i)
            blk.eliminate_zeros()
            parts.append(_topk_rows(blk, self.cfg.passage_graph_top_k, self.cfg.passage_graph_min_weight))
        similarity = sp.vstack(parts, format="csr") if parts else csr_matrix((n_rows, n_rows))
        self._passage_graph = similarity.maximum(similarity.T).tocsr()

    def _build_bridge_lookups(self) -> None:
        mention_csc = self._mention.tocsc()
        self._entity_to_sentences = [
            mention_csc.indices[mention_csc.indptr[e]:mention_csc.indptr[e + 1]].copy()
            for e in range(mention_csc.shape[1])
        ]
        self._sentence_to_entities = [
            self._mention.indices[self._mention.indptr[s]:self._mention.indptr[s + 1]].copy()
            for s in range(self._mention.shape[0])
        ]

    # ------------------------------------------------------------------ online: retrieval

    def encode_queries(self, questions: list[str]) -> np.ndarray:
        return _row_normalize_dense(np.asarray(
            self.project.embedding_model.encode(questions, normalize_embeddings=True, show_progress_bar=False,
                                                batch_size=self.project.batch_size),
            dtype=np.float32))

    def score_query(self, question: str, query_embedding: np.ndarray) -> dict[str, Any]:
        """Stages 1-2 and score fusion for one query; the reranker consumes this."""
        dense_scores = self._passage_embs @ query_embedding
        seeds = self._link_query_entities(question, query_embedding)
        active, hops, parents = self._bridge_propagation(query_embedding, seeds)
        candidates = self._collect_candidates(dense_scores, active)
        ppr_scores, coverage_scores = self._subgraph_ppr(candidates, dense_scores, active)
        fused = self._fuse_scores(candidates, dense_scores, ppr_scores, coverage_scores)
        return {"dense_scores": dense_scores, "seeds": seeds, "active": active, "hops": hops,
                "parents": parents, "candidates": candidates, "ppr": ppr_scores,
                "coverage": coverage_scores, "fused": fused}

    def retrieve(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fused ranking without the learned reranker (ablation "no_rerank")."""
        if not self._indexed:
            raise RuntimeError("Call index(passages) before retrieve(questions)")
        if not questions:
            return []
        embeddings = self.encode_queries([str(row["question"]) for row in questions])
        results = []
        for row, query_embedding in tqdm(zip(questions, embeddings), total=len(questions), desc="Retrieving(SpearRAG)"):
            s = self.score_query(str(row["question"]), query_embedding)
            order = np.argsort(s["fused"])[::-1]
            ranked, ranked_scores = s["candidates"][order], s["fused"][order]
            k = min(self.cfg.retrieval_top_k, len(ranked))
            top = ranked[:k]
            ordered = (self._order_for_generation(ranked, ranked_scores, s["dense_scores"], s["active"])[:k]
                       if self.cfg.enable_chain_ordering else top)
            active_idx = np.flatnonzero(s["active"] > 0.0)
            active_idx = active_idx[np.argsort(s["active"][active_idx])[::-1]]
            results.append({
                "question": str(row["question"]),
                "gold_answer": row.get("answer", ""),
                "sorted_passage": [self._passage_texts[i] for i in top],
                "sorted_passage_scores": [float(v) for v in ranked_scores[:k]],
                "ordered_passage": [self._passage_texts[i] for i in ordered],
                "diagnostics": {
                    "seed_entities": [{"entity": self._entity_texts[i], "score": float(v)}
                                      for i, v in sorted(s["seeds"].items(), key=lambda x: -x[1])],
                    "activated_entities": [{"entity": self._entity_texts[i], "score": float(s["active"][i]),
                                            "hop": int(s["hops"][i])} for i in active_idx[:24]],
                    "bridge_parents": self._parent_diagnostics(s["parents"], active_idx[:24]),
                    "candidate_passages": int(len(s["candidates"])),
                },
            })
        return results

    # -- (1) Entity Bridging --------------------------------------------------------------

    def _link_query_entities(self, question: str, query_embedding: np.ndarray) -> dict[int, float]:
        """Entity Seeding: exact (case-folded) match first, then embedding match for the rest."""
        mentions = list(dict.fromkeys(self.ner.question_ner(question)))
        if not self._entity_hids or not mentions:
            return {}
        linked: dict[int, float] = {}
        unresolved: list[str] = []
        for mention in mentions:
            exact = self._entity_casefold_to_indices.get(str(mention).casefold(), [])
            if exact:
                for index in exact:
                    contextual = max(0.0, float(self._entity_embs[index] @ query_embedding))
                    linked[index] = max(linked.get(index, 0.0), 0.8 + 0.2 * contextual)
            else:
                unresolved.append(str(mention))
        if unresolved:
            mention_embeddings = _row_normalize_dense(np.asarray(
                self.project.embedding_model.encode(unresolved, normalize_embeddings=True, show_progress_bar=False,
                                                    batch_size=self.project.batch_size), dtype=np.float32))
            similarities = self._entity_embs @ mention_embeddings.T
            for column in range(similarities.shape[1]):
                scores = similarities[:, column]
                if len(scores) == 0:
                    continue
                count = min(self.cfg.seed_candidates_per_mention, len(scores))
                candidate = np.argpartition(scores, -count)[-count:]
                candidate = candidate[np.argsort(scores[candidate])[::-1]]
                best_score = float(scores[candidate[0]])
                for index in candidate:
                    score = float(scores[index])
                    if score < self.cfg.seed_similarity_threshold:
                        continue
                    if best_score - score > self.cfg.seed_similarity_margin:
                        continue
                    contextual = max(0.0, float(self._entity_embs[index] @ query_embedding))
                    combined = 0.75 * score + 0.25 * contextual
                    linked[int(index)] = max(linked.get(int(index), 0.0), combined)
        return linked

    def _bridge_propagation(self, query_embedding: np.ndarray, seeds: dict[int, float]
                            ) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int]]]:
        """Bridge Propagation: activate entities hop by hop through query-relevant sentences."""
        n_entities = len(self._entity_hids)
        active = np.zeros(n_entities, dtype=np.float64)
        hops = np.full(n_entities, -1, dtype=np.int16)
        parents: dict[int, tuple[int, int]] = {}
        frontier: dict[int, float] = {}
        for entity, score in seeds.items():
            active[entity] = max(active[entity], float(score))
            hops[entity] = 0
            frontier[entity] = active[entity]

        expanded_at = np.zeros(n_entities, dtype=np.float64)
        for hop in range(1, self.cfg.max_bridge_hops + 1):
            next_frontier: dict[int, float] = {}
            for entity, entity_score in sorted(frontier.items(), key=lambda item: -item[1]):
                if entity_score <= expanded_at[entity] + 1e-12:
                    continue
                expanded_at[entity] = entity_score
                sentence_indices = self._entity_to_sentences[entity]
                if sentence_indices.size == 0:
                    continue
                similarities = self._sentence_embs[sentence_indices] @ query_embedding
                count = min(self.cfg.sentences_per_entity, len(sentence_indices))
                local_top = np.argpartition(similarities, -count)[-count:]
                local_top = local_top[np.argsort(similarities[local_top])[::-1]]
                for local_position in local_top:
                    raw_similarity = float(similarities[local_position])
                    if raw_similarity < self.cfg.sentence_similarity_floor:
                        continue
                    sentence = int(sentence_indices[local_position])
                    calibrated = (raw_similarity - self.cfg.sentence_similarity_floor) / max(
                        1.0 - self.cfg.sentence_similarity_floor, 1e-6)
                    next_entities = self._sentence_to_entities[sentence]
                    degree_penalty = math.sqrt(max(len(next_entities) - 1, 1))
                    propagated = entity_score * calibrated * self.cfg.hop_decay / degree_penalty
                    if propagated < self.cfg.activation_threshold:
                        continue
                    for next_entity in next_entities:
                        next_entity = int(next_entity)
                        if next_entity == entity:
                            continue
                        if propagated > active[next_entity] + 1e-12:
                            active[next_entity] = propagated
                            hops[next_entity] = hop
                            parents[next_entity] = (entity, sentence)
                            next_frontier[next_entity] = max(next_frontier.get(next_entity, 0.0), propagated)
            if not next_frontier:
                break
            if np.count_nonzero(active) > self.cfg.max_active_entities:
                keep = np.argpartition(active, -self.cfg.max_active_entities)[-self.cfg.max_active_entities:]
                keep_mask = np.zeros(n_entities, dtype=bool)
                keep_mask[keep] = True
                active[~keep_mask] = 0.0
                next_frontier = {e: s for e, s in next_frontier.items() if keep_mask[e]}
            frontier = next_frontier
        return active, hops, parents

    def _collect_candidates(self, dense_scores: np.ndarray, active_entities: np.ndarray) -> np.ndarray:
        """Bridged Candidate Generation: dense top-N U bridge passages U passage-graph neighbours."""
        candidates: set[int] = set()
        dense_count = min(self.cfg.global_dense_candidates, len(dense_scores))
        candidates.update(map(int, np.argpartition(dense_scores, -dense_count)[-dense_count:]))

        active_index = np.flatnonzero(active_entities > 0.0)
        if active_index.size and self.cfg.enable_bridge_candidates:
            bridge_signal = np.asarray(self._contain_bin[:, active_index] @ active_entities[active_index]).ravel()
            bridge_hits = np.flatnonzero(bridge_signal > 0.0)
            if len(bridge_hits) > self.cfg.max_bridge_passages:
                local = np.argpartition(bridge_signal[bridge_hits], -self.cfg.max_bridge_passages)[
                    -self.cfg.max_bridge_passages:]
                bridge_hits = bridge_hits[local]
            candidates.update(map(int, bridge_hits))

        if self.cfg.passage_neighbor_expansion > 0 and candidates:
            expanded: set[int] = set()
            for passage in np.fromiter(candidates, dtype=np.int32):
                start, end = self._passage_graph.indptr[passage], self._passage_graph.indptr[passage + 1]
                neighbors = self._passage_graph.indices[start:end]
                weights = self._passage_graph.data[start:end]
                count = min(self.cfg.passage_neighbor_expansion, len(neighbors))
                if count:
                    expanded.update(map(int, neighbors[np.argpartition(weights, -count)[-count:]]))
            candidates.update(expanded)

        indices = np.fromiter(candidates, dtype=np.int32)
        if len(indices) > self.cfg.max_candidate_passages:
            if active_index.size:
                coverage = np.asarray(self._contain_bin[indices][:, active_index]
                                      @ active_entities[active_index]).ravel()
            else:
                coverage = np.zeros(len(indices), dtype=np.float64)
            priority = dense_scores[indices] + 0.25 * coverage
            indices = indices[np.argpartition(priority, -self.cfg.max_candidate_passages)[
                -self.cfg.max_candidate_passages:]]
        return np.sort(indices)

    # -- (2) Subgraph Propagation -------------------------------------------------------

    def _global_ppr_adjacency(self) -> csr_matrix:
        """Full passage+entity graph with the same edge construction (ppr_scope="global" control)."""
        if self._global_adjacency is None:
            contain = self._contain.tocsr().astype(np.float64)
            n_entities = contain.shape[1]
            passage_graph = (self.cfg.passage_graph_edge_weight * self._passage_graph).tocsr()
            self._global_adjacency = sp.bmat(
                [[passage_graph, contain], [contain.T, csr_matrix((n_entities, n_entities), dtype=np.float64)]],
                format="csr")
        return self._global_adjacency

    def _subgraph_ppr(self, passages: np.ndarray, dense_scores: np.ndarray, active_entities: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Subgraph PPR on the query-induced subgraph; returns (PPR, entity coverage) over the candidates."""
        entity_indices = np.unique(self._contain[passages].indices)
        active_index = np.flatnonzero(active_entities > 0.0)
        entity_indices = np.union1d(entity_indices, active_index).astype(np.int32)
        contain = self._contain[passages][:, entity_indices].tocoo()
        passage_entity = csr_matrix((contain.data.astype(np.float64, copy=True), (contain.row, contain.col)),
                                    shape=(len(passages), len(entity_indices)))
        local_passage_graph = (self.cfg.passage_graph_edge_weight
                               * self._passage_graph[passages][:, passages]).tocsr()
        adjacency = sp.bmat(
            [[local_passage_graph, passage_entity],
             [passage_entity.T, csr_matrix((len(entity_indices), len(entity_indices)), dtype=np.float64)]],
            format="csr")

        # Restart: dense top-n passages (softmax) and activated entities.
        passage_restart = np.zeros(len(passages), dtype=np.float64)
        dense_count = min(self.cfg.dense_restart_top_n, len(passages))
        local_dense_top = np.argpartition(dense_scores[passages], -dense_count)[-dense_count:]
        passage_restart[local_dense_top] = _softmax_sparse_values(
            dense_scores[passages[local_dense_top]], self.cfg.dense_temperature)
        entity_restart = _l1(active_entities[entity_indices].astype(np.float64))
        entity_mix = self.cfg.restart_entity_weight if entity_restart.sum() > 0.0 else 0.0
        restart = np.concatenate([(1.0 - entity_mix) * _l1(passage_restart), entity_mix * entity_restart])

        if self.cfg.enable_subgraph_ppr and self.cfg.ppr_scope == "global":
            global_adjacency = self._global_ppr_adjacency()
            n_passages = self._passage_graph.shape[0]
            global_restart = np.zeros(global_adjacency.shape[0], dtype=np.float64)
            global_restart[passages] = restart[:len(passages)]
            global_restart[n_passages + entity_indices] = restart[len(passages):]
            ppr = sparse_ppr(global_adjacency, global_restart, self.cfg.ppr_damping,
                             self.cfg.ppr_tolerance, self.cfg.ppr_max_iterations)[passages]
        elif self.cfg.enable_subgraph_ppr:
            ppr = sparse_ppr(adjacency, restart, self.cfg.ppr_damping,
                             self.cfg.ppr_tolerance, self.cfg.ppr_max_iterations)
        else:
            ppr = restart   # no propagation: the restart distribution is the score

        if active_index.size:
            coverage = np.asarray(self._contain_bin[passages][:, active_index] @ active_entities[active_index]).ravel()
        else:
            coverage = np.zeros(len(passages), dtype=np.float64)
        return _l1(ppr[:len(passages)]), _l1(coverage)

    # -- (3) Score fusion ----------------------------------------------------------------

    def _fuse_scores(self, passages: np.ndarray, dense_scores: np.ndarray, ppr_scores: np.ndarray,
                     coverage_scores: np.ndarray) -> np.ndarray:
        dense_part = _softmax_sparse_values(dense_scores[passages], self.cfg.dense_temperature)
        parts = [ppr_scores, dense_part, coverage_scores]
        weights = np.asarray([self.cfg.fusion_ppr_weight, self.cfg.fusion_dense_weight,
                              self.cfg.fusion_coverage_weight], dtype=np.float64)
        weights = _l1(weights * np.asarray([part.sum() > 0.0 for part in parts]))
        return _l1(np.asarray(sum(w * part for w, part in zip(weights, parts))))

    # -- generator-side evidence order (fused ranking only) ------------------------------

    def _order_for_generation(self, ranked_passages: np.ndarray, ranked_scores: np.ndarray,
                              dense_scores: np.ndarray, active_entities: np.ndarray) -> np.ndarray:
        """Greedy chain order: prefer passages sharing entities with what is already chosen."""
        pool_size = min(self.cfg.generation_pool_size, len(ranked_passages))
        pool = ranked_passages[:pool_size].copy()
        base = _l1(ranked_scores[:pool_size])
        if pool_size <= 1:
            return pool
        rows = self._contain_bin[pool]
        selected_local = [0]
        remaining = set(range(1, pool_size))
        frontier_entities = set(rows[0].indices.tolist())
        while remaining:
            best_position, best_value = None, -math.inf
            for position in remaining:
                entities = set(rows[position].indices.tolist())
                connectivity = len(entities & frontier_entities) / math.sqrt(max(len(entities), 1))
                active_coverage = float(active_entities[list(entities)].sum() if entities else 0.0)
                value = ((1.0 - self.cfg.generation_connectivity_weight) * base[position]
                         + self.cfg.generation_connectivity_weight * connectivity
                         + 0.05 * active_coverage
                         + 0.02 * max(0.0, float(dense_scores[pool[position]])))
                if value > best_value:
                    best_value, best_position = value, position
            selected_local.append(best_position)
            remaining.remove(best_position)
            frontier_entities.update(rows[best_position].indices.tolist())
        ordered = pool[np.asarray(selected_local, dtype=np.int32)]
        return np.concatenate([ordered, ranked_passages[pool_size:]])

    def _parent_diagnostics(self, parents: dict[int, tuple[int, int]], entity_indices: Iterable[int]
                            ) -> list[dict[str, str]]:
        rows = []
        for entity in entity_indices:
            if int(entity) in parents:
                parent, sentence = parents[int(entity)]
                rows.append({"from": self._entity_texts[parent], "sentence": self._sentence_texts[sentence],
                             "to": self._entity_texts[int(entity)]})
        return rows
