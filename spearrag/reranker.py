"""SpearRAG stage 3 -- Adaptive Reranking: a graph-aware listwise reranker over the fused candidates.

For each query the retriever's candidates are cut to the top ``candidate_limit`` by fused score, and a
small transformer scores them jointly:

* per-passage input: query/passage embeddings + 11 scalar features (dense, Subgraph PPR, entity coverage,
  fused score, activation mass by hop, local subgraph degree)
* pairwise input (attention bias): semantic similarity, passage-graph edge, active-entity Jaccard,
  entity Jaccard
* output: a unary relevance logit per passage and a pairwise compatibility logit; the final top-k is a
  greedy selection on unary + ``pair_selection_weight`` x compatibility.

Training: listwise softmax cross-entropy on gold passages (the paper's setting; the binary / margin / pair
loss terms exist but default to weight 0), AdamW, early stopping on validation Complete@k.

Provenance: the research code's ``qacf_listwise_reranker.py`` run with
``community_ablation="minus_reranker_community_features"``. That setting fed three community-derived
inputs as constant zeros (scalar ``community_route_z``, ``inverse_community_size``; pair
``same_community``); here those columns are removed, which leaves the function the network computes
unchanged (zero inputs contribute nothing). See README "Reproducibility notes".
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spearrag.retriever import SpearRAGRetriever

logger = logging.getLogger(__name__)

SCALAR_FEATURE_NAMES = (
    "dense_z",
    "dense_rank",
    "log_ppr_z",
    "ppr_rank",
    "coverage",
    "fused_score_z",
    "active_entity_mass",
    "seed_entity_mass",
    "hop1_entity_mass",
    "hop2plus_entity_mass",
    "local_graph_degree",
)
PAIR_FEATURE_NAMES = (
    "semantic_similarity",
    "passage_graph_edge",
    "active_entity_jaccard",
    "entity_jaccard",
)
# Graph-derived scalar features (ablation "no_graph_features" zeroes exactly these; the pairwise
# attention bias stays on).
GRAPH_SCALAR_FEATURE_NAMES = (
    "log_ppr_z",
    "ppr_rank",
    "active_entity_mass",
    "seed_entity_mass",
    "hop1_entity_mass",
    "hop2plus_entity_mass",
    "local_graph_degree",
)
assert set(GRAPH_SCALAR_FEATURE_NAMES) <= set(SCALAR_FEATURE_NAMES)

GoldResolver = Callable[[dict[str, Any], SpearRAGRetriever], set[int]]


@dataclass
class RerankerConfig:
    """Model, objective and training hyper-parameters (paper defaults)."""

    candidate_limit: int = 128        # passages the reranker sees (top by fused score)
    retrieval_top_k: int = 5
    force_gold_during_training: bool = True

    # Network
    hidden_dim: int = 192
    num_heads: int = 6
    num_layers: int = 2
    feedforward_multiplier: int = 3
    dropout: float = 0.10

    # Loss (paper: listwise only)
    listwise_loss_weight: float = 1.00
    binary_loss_weight: float = 0.0
    complete_margin_loss_weight: float = 0.0
    pair_loss_weight: float = 0.0
    complete_margin: float = 0.50

    # Greedy selection: weight of learned pair compatibility after the first pick
    pair_selection_weight: float = 0.35

    # Optimiser
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # "full" or "minus_graph_scalars" (ablation "no_graph_features")
    FEATURE_ABLATIONS = ("full", "minus_graph_scalars")
    feature_ablation: str = "full"

    def validate(self) -> None:
        if self.candidate_limit < self.retrieval_top_k:
            raise ValueError("candidate_limit must be >= retrieval_top_k")
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive")
        if self.hidden_dim < 8 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be >= 8 and divisible by num_heads")
        if self.num_layers < 1 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("num_layers, batch_size and epochs must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.feature_ablation not in self.FEATURE_ABLATIONS:
            raise ValueError(f"feature_ablation must be one of {self.FEATURE_ABLATIONS}")


@dataclass
class CandidateExample:
    """One variable-length listwise instance."""

    question: str
    passage_indices: np.ndarray
    query_embedding: np.ndarray
    passage_embeddings: np.ndarray
    scalar_features: np.ndarray
    pair_features: np.ndarray
    baseline_scores: np.ndarray
    gold_mask: np.ndarray
    full_candidate_count: int
    candidate_gold_count: int
    total_gold_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_complete(self) -> bool:
        return self.total_gold_count > 0 and self.candidate_gold_count == self.total_gold_count

    @property
    def visible_candidate_complete(self) -> bool:
        return self.total_gold_count > 0 and int(self.gold_mask.sum()) == self.total_gold_count


# ----------------------------------------------------------------------------- feature helpers

def _safe_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    values = np.where(finite, values, float(np.median(values[finite])))
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.mean()) / std).astype(np.float32)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    values = np.where(np.isfinite(values), values, 0.0)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def _rank_score(values: np.ndarray) -> np.ndarray:
    """1 for the best item, 0 for the worst."""
    values = np.asarray(values)
    count = len(values)
    if count <= 1:
        return np.ones(count, dtype=np.float32)
    order = np.argsort(values, kind="stable")[::-1]
    ranks = np.empty(count, dtype=np.float32)
    ranks[order] = np.arange(count, dtype=np.float32)
    return 1.0 - ranks / float(count - 1)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def doc_index_gold_resolver(doc_id_to_index: dict[Any, int]) -> GoldResolver:
    """Gold resolver for datasets whose queries list ``gold_doc_ids`` of the indexed documents."""
    def resolve(row: dict[str, Any], _retriever: SpearRAGRetriever) -> set[int]:
        return {doc_id_to_index[d] for d in row.get("gold_doc_ids", []) if d in doc_id_to_index}
    return resolve


class FeatureExtractor:
    """Run SpearRAG stages 1-2 + fusion for a query and build the reranker's input."""

    def __init__(self, retriever: SpearRAGRetriever, config: RerankerConfig) -> None:
        config.validate()
        if not retriever._indexed:
            raise RuntimeError("index the corpus before extracting features")
        self.retriever = retriever
        self.config = config

    def encode_queries(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.empty((0, int(self.retriever._passage_embs.shape[1])), dtype=np.float32)
        embeddings = self.retriever.project.embedding_model.encode(
            [str(row["question"]) for row in rows], normalize_embeddings=True, show_progress_bar=False,
            batch_size=self.retriever.project.batch_size)
        return _normalize_rows(np.asarray(embeddings, dtype=np.float32))

    def build(self, row: dict[str, Any], query_embedding: np.ndarray,
              gold_indices: Optional[set[int]] = None, training: bool = False) -> CandidateExample:
        r = self.retriever
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        question = str(row["question"])
        s = r.score_query(question, query_embedding)
        full_passages, dense_scores = s["candidates"], s["dense_scores"]

        gold_indices = set(gold_indices or set())
        selected_local = self._prefilter(full_passages, s["fused"], gold_indices,
                                         force_gold=training and self.config.force_gold_during_training)
        passages = full_passages[selected_local]
        ppr, coverage, base = s["ppr"][selected_local], s["coverage"][selected_local], s["fused"][selected_local]

        scalar = self._scalar_features(passages, dense_scores, ppr, coverage, base, s["active"], s["hops"])
        pair = self._pair_features(passages, s["active"])
        return CandidateExample(
            question=question,
            passage_indices=passages.astype(np.int32, copy=False),
            query_embedding=query_embedding.astype(np.float32, copy=False),
            passage_embeddings=r._passage_embs[passages].astype(np.float32, copy=False),
            scalar_features=scalar,
            pair_features=pair,
            baseline_scores=base.astype(np.float32, copy=False),
            gold_mask=np.asarray([int(i) in gold_indices for i in passages], dtype=bool),
            full_candidate_count=int(len(full_passages)),
            candidate_gold_count=int(sum(int(i) in gold_indices for i in full_passages)),
            total_gold_count=int(len(gold_indices)),
            metadata={"seed_count": int(len(s["seeds"])),
                      "active_entity_count": int(np.count_nonzero(s["active"])),
                      "max_propagation_hop": int(s["hops"].max()) if s["hops"].size else -1},
        )

    def _prefilter(self, full_passages: np.ndarray, baseline_scores: np.ndarray,
                   gold_indices: set[int], force_gold: bool) -> np.ndarray:
        """Top ``candidate_limit`` by fused score; during training gold passages are always kept."""
        order = np.argsort(baseline_scores, kind="stable")[::-1]
        limit = min(self.config.candidate_limit, len(order))
        if not force_gold or not gold_indices:
            return order[:limit]
        passage_to_local = {int(p): local for local, p in enumerate(full_passages)}
        gold_local = [passage_to_local[i] for i in gold_indices if i in passage_to_local]
        gold_local.sort(key=lambda local: -float(baseline_scores[local]))
        gold_set = set(gold_local[:limit])
        non_gold = [int(local) for local in order if int(local) not in gold_set]
        chosen = list(gold_set) + non_gold[:max(0, limit - len(gold_set))]
        chosen.sort(key=lambda local: -float(baseline_scores[local]))
        return np.asarray(chosen, dtype=np.int32)

    def _scalar_features(self, passages, dense_scores, ppr, coverage, base, active, hops) -> np.ndarray:
        r = self.retriever
        active_index = np.flatnonzero(active > 0.0)
        if active_index.size:
            active_mass = np.asarray(r._contain_bin[passages][:, active_index] @ active[active_index]).ravel()

            def hop_mass(mask: np.ndarray) -> np.ndarray:
                indices = active_index[mask]
                if indices.size == 0:
                    return np.zeros(len(passages), dtype=np.float64)
                return np.asarray(r._contain_bin[passages][:, indices] @ active[indices]).ravel()

            active_hops = hops[active_index]
            seed_mass, hop1_mass, hop2plus_mass = (hop_mass(active_hops == 0), hop_mass(active_hops == 1),
                                                   hop_mass(active_hops >= 2))
        else:
            active_mass = np.zeros(len(passages), dtype=np.float64)
            seed_mass, hop1_mass, hop2plus_mass = active_mass.copy(), active_mass.copy(), active_mass.copy()
        local_degree = np.asarray(r._passage_graph[passages][:, passages].sum(axis=1)).ravel()

        result = np.column_stack((
            _safe_zscore(dense_scores[passages]),
            _rank_score(dense_scores[passages]),
            _safe_zscore(np.log1p(np.maximum(ppr, 0.0) * len(ppr))),
            _rank_score(ppr),
            _minmax(coverage),
            _safe_zscore(base),
            _minmax(active_mass),
            _minmax(seed_mass),
            _minmax(hop1_mass),
            _minmax(hop2plus_mass),
            _minmax(local_degree),
        )).astype(np.float32)
        assert result.shape[1] == len(SCALAR_FEATURE_NAMES), "scalar feature schema mismatch"
        if self.config.feature_ablation == "minus_graph_scalars":
            for name in GRAPH_SCALAR_FEATURE_NAMES:
                result[:, SCALAR_FEATURE_NAMES.index(name)] = 0.0
        return result

    def _pair_features(self, passages: np.ndarray, active: np.ndarray) -> np.ndarray:
        r = self.retriever
        embeddings = r._passage_embs[passages]
        semantic = np.clip((embeddings @ embeddings.T + 1.0) * 0.5, 0.0, 1.0)

        graph = r._passage_graph[passages][:, passages].toarray().astype(np.float32)
        graph_max = float(graph.max()) if graph.size else 0.0
        if graph_max > 0.0:
            graph /= graph_max

        contain = r._contain_bin[passages].astype(np.float32)
        intersection = (contain @ contain.T).toarray().astype(np.float32)
        entity_degree = np.asarray(contain.sum(axis=1)).ravel()
        union = entity_degree[:, None] + entity_degree[None, :] - intersection
        entity_jaccard = np.divide(intersection, np.maximum(union, 1e-8),
                                   out=np.zeros_like(intersection), where=union > 0.0)

        active_index = np.flatnonzero(active > 0.0)
        if active_index.size:
            active_weight = active[active_index].astype(np.float64)
            active_weight /= max(float(active_weight.sum()), 1e-12)
            presence = r._contain_bin[passages][:, active_index].toarray() > 0.0
            weighted_presence = presence.astype(np.float64) * active_weight[None, :]
            active_intersection = weighted_presence @ presence.T.astype(np.float64)
            active_mass = weighted_presence.sum(axis=1)
            active_union = active_mass[:, None] + active_mass[None, :] - active_intersection
            active_jaccard = np.divide(active_intersection, np.maximum(active_union, 1e-12),
                                       out=np.zeros_like(active_intersection), where=active_union > 0.0)
        else:
            active_jaccard = np.zeros_like(semantic)

        result = np.stack([semantic, graph, active_jaccard, entity_jaccard], axis=-1).astype(np.float32)
        assert result.shape[-1] == len(PAIR_FEATURE_NAMES), "pair feature schema mismatch"
        return result


def build_examples(extractor: FeatureExtractor, rows: Sequence[dict[str, Any]], training: bool,
                   gold_resolver: GoldResolver) -> tuple[list[CandidateExample], dict[str, float | int]]:
    """Precompute features once so training epochs do not repeat retrieval."""
    query_embeddings = extractor.encode_queries(rows)
    examples, unresolved, full_complete, truncated_complete = [], 0, 0, 0
    for row, query_embedding in zip(rows, query_embeddings):
        gold = gold_resolver(row, extractor.retriever)
        unresolved += int(not gold)
        example = extractor.build(row, query_embedding, gold_indices=gold, training=training)
        examples.append(example)
        full_complete += int(example.candidate_complete)
        truncated_complete += int(example.visible_candidate_complete)
    resolved = len(rows) - unresolved
    return examples, {"rows": len(rows), "resolved_gold_rows": resolved, "unresolved_gold_rows": unresolved,
                      "full_candidate_complete_recall": full_complete / max(resolved, 1),
                      "truncated_candidate_complete_recall": truncated_complete / max(resolved, 1)}


def collate_examples(examples: Sequence[CandidateExample], device) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(examples)
    max_count = max(len(e.passage_indices) for e in examples)
    embedding_dim = examples[0].passage_embeddings.shape[1]
    scalar_dim = examples[0].scalar_features.shape[1]
    pair_dim = examples[0].pair_features.shape[2]
    query = np.zeros((batch_size, embedding_dim), dtype=np.float32)
    passages = np.zeros((batch_size, max_count, embedding_dim), dtype=np.float32)
    scalar = np.zeros((batch_size, max_count, scalar_dim), dtype=np.float32)
    pair = np.zeros((batch_size, max_count, max_count, pair_dim), dtype=np.float32)
    mask = np.zeros((batch_size, max_count), dtype=bool)
    gold = np.zeros((batch_size, max_count), dtype=bool)
    baseline = np.zeros((batch_size, max_count), dtype=np.float32)
    for b, e in enumerate(examples):
        n = len(e.passage_indices)
        query[b] = e.query_embedding
        passages[b, :n] = e.passage_embeddings
        scalar[b, :n] = e.scalar_features
        pair[b, :n, :n] = e.pair_features
        mask[b, :n] = True
        gold[b, :n] = e.gold_mask
        baseline[b, :n] = e.baseline_scores
    return {k: torch.as_tensor(v, device=device) for k, v in
            dict(query=query, passages=passages, scalar=scalar, pair=pair, mask=mask, gold=gold,
                 baseline=baseline).items()}


# ----------------------------------------------------------------------------- model

class EdgeBiasedSelfAttention(nn.Module):
    """Self-attention over the candidate set, biased by pairwise graph features."""

    def __init__(self, hidden_dim: int, num_heads: int, pair_feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.edge_bias = nn.Linear(pair_feature_dim, num_heads, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, pair_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, count, hidden = nodes.shape
        query, key, value = self.qkv(nodes).view(batch, count, 3, self.num_heads, self.head_dim).unbind(dim=2)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits + self.edge_bias(pair_features).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~mask[:, None, None, :], -1e4)
        attention = self.dropout(torch.softmax(logits, dim=-1))
        context = torch.matmul(attention, value).transpose(1, 2).reshape(batch, count, hidden)
        return self.output(context) * mask.unsqueeze(-1)


class GraphListwiseBlock(nn.Module):
    def __init__(self, config: RerankerConfig, pair_feature_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attention = EdgeBiasedSelfAttention(config.hidden_dim, config.num_heads, pair_feature_dim,
                                                 config.dropout)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        ff = config.hidden_dim * config.feedforward_multiplier
        self.feedforward = nn.Sequential(nn.Linear(config.hidden_dim, ff), nn.GELU(), nn.Dropout(config.dropout),
                                         nn.Linear(ff, config.hidden_dim), nn.Dropout(config.dropout))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, nodes, pair_features, mask):
        nodes = nodes + self.dropout(self.attention(self.norm1(nodes), pair_features, mask))
        nodes = nodes + self.feedforward(self.norm2(nodes))
        return nodes * mask.unsqueeze(-1)


class GraphAwareListwiseReranker(nn.Module):
    """Joint candidate encoder with unary relevance and pairwise compatibility heads."""

    def __init__(self, embedding_dim: int, config: RerankerConfig,
                 scalar_feature_dim: int = len(SCALAR_FEATURE_NAMES),
                 pair_feature_dim: int = len(PAIR_FEATURE_NAMES)) -> None:
        super().__init__()
        config.validate()
        self.embedding_dim = int(embedding_dim)
        self.config = copy.deepcopy(config)
        self.semantic_projection = nn.Sequential(nn.Linear(4 * embedding_dim, config.hidden_dim), nn.GELU(),
                                                 nn.Dropout(config.dropout))
        self.scalar_projection = nn.Sequential(nn.Linear(scalar_feature_dim, config.hidden_dim), nn.GELU())
        self.input_norm = nn.LayerNorm(config.hidden_dim)
        self.blocks = nn.ModuleList([GraphListwiseBlock(config, pair_feature_dim) for _ in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.unary_head = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.GELU(),
                                        nn.Dropout(config.dropout), nn.Linear(config.hidden_dim // 2, 1))
        self.pair_query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.pair_key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.pair_edge = nn.Linear(pair_feature_dim, 1)

    def forward(self, query, passages, scalar, pair, mask):
        expanded_query = query[:, None, :].expand_as(passages)
        semantic = torch.cat([passages, expanded_query, passages * expanded_query,
                              torch.abs(passages - expanded_query)], dim=-1)
        nodes = self.semantic_projection(semantic) + self.scalar_projection(scalar)
        nodes = self.input_norm(nodes) * mask.unsqueeze(-1)
        for block in self.blocks:
            nodes = block(nodes, pair, mask)
        nodes = self.output_norm(nodes)
        unary = self.unary_head(nodes).squeeze(-1).masked_fill(~mask, -1e4)
        pair_query, pair_key = self.pair_query(nodes), self.pair_key(nodes)
        pair_logits = torch.matmul(pair_query, pair_key.transpose(-2, -1)) / math.sqrt(pair_query.shape[-1])
        pair_logits = pair_logits + self.pair_edge(pair).squeeze(-1)
        pair_logits = 0.5 * (pair_logits + pair_logits.transpose(-2, -1))
        valid_pair = mask[:, :, None] & mask[:, None, :]
        return unary, pair_logits.masked_fill(~valid_pair, -1e4)

    @torch.no_grad()
    def score_examples(self, examples: Sequence[CandidateExample], device: Optional[str] = None
                       ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(selection order, unary logits, pair logits) per example."""
        if not examples:
            return []
        target = torch.device(device or self.config.device)
        self.to(target)
        self.eval()
        batch = collate_examples(examples, target)
        unary, pair = self(batch["query"], batch["passages"], batch["scalar"], batch["pair"], batch["mask"])
        outputs = []
        for i, example in enumerate(examples):
            n = len(example.passage_indices)
            unary_np, pair_np = unary[i, :n].cpu().numpy(), pair[i, :n, :n].cpu().numpy()
            outputs.append((greedy_set_order(unary_np, pair_np, self.config.pair_selection_weight),
                            unary_np, pair_np))
        return outputs


def create_reranker(retriever: SpearRAGRetriever, config: RerankerConfig) -> GraphAwareListwiseReranker:
    if not retriever._indexed:
        raise RuntimeError("index the corpus before creating the reranker")
    return GraphAwareListwiseReranker(int(retriever._passage_embs.shape[1]), config)


def reranker_loss(unary_logits, pair_logits, gold, mask, config: RerankerConfig):
    valid_rows = gold.any(dim=1)
    if not valid_rows.any():
        zero = unary_logits.sum() * 0.0
        return zero, {"listwise": 0.0, "binary": 0.0, "complete_margin": 0.0, "pair": 0.0}

    masked_logits = unary_logits.masked_fill(~mask, -1e4)
    target = gold.float()
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    listwise = (-(target * F.log_softmax(masked_logits, dim=1)).sum(dim=1))[valid_rows].mean()

    valid_logits, valid_targets = unary_logits[mask], gold[mask].float()
    positive_weight = ((1.0 - valid_targets).sum().clamp_min(1.0) / valid_targets.sum().clamp_min(1.0)).clamp(max=30.0)
    binary = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, pos_weight=positive_weight)

    margin_terms, pair_positive, pair_negative = [], [], []
    for row in range(unary_logits.shape[0]):
        positives = gold[row] & mask[row]
        negatives = (~gold[row]) & mask[row]
        if positives.any() and negatives.any():
            hardest = unary_logits[row, negatives].max()
            margin_terms.append(F.relu(config.complete_margin + hardest - unary_logits[row, positives]).mean())
        valid_pair = mask[row, :, None] & mask[row, None, :]
        diagonal = torch.eye(mask.shape[1], device=mask.device, dtype=torch.bool)
        positive_pair = gold[row, :, None] & gold[row, None, :] & valid_pair & ~diagonal
        negative_pair = (gold[row, :, None] ^ gold[row, None, :]) & valid_pair
        if positive_pair.any():
            pair_positive.append(F.softplus(-pair_logits[row][positive_pair]).mean())
        if negative_pair.any():
            pair_negative.append(F.softplus(pair_logits[row][negative_pair]).mean())
    complete_margin = torch.stack(margin_terms).mean() if margin_terms else unary_logits.sum() * 0.0
    pair_parts = pair_positive + pair_negative
    pair_loss = torch.stack(pair_parts).mean() if pair_parts else unary_logits.sum() * 0.0

    total = (config.listwise_loss_weight * listwise + config.binary_loss_weight * binary
             + config.complete_margin_loss_weight * complete_margin + config.pair_loss_weight * pair_loss)
    return total, {"listwise": float(listwise.detach().cpu()), "binary": float(binary.detach().cpu()),
                   "complete_margin": float(complete_margin.detach().cpu()), "pair": float(pair_loss.detach().cpu())}


def greedy_set_order(unary_logits: np.ndarray, pair_logits: np.ndarray, pair_selection_weight: float) -> np.ndarray:
    """Greedy order on unary relevance + max compatibility with already-selected passages."""
    unary_logits = np.asarray(unary_logits, dtype=np.float64)
    pair_logits = np.asarray(pair_logits, dtype=np.float64)
    count = len(unary_logits)
    if count == 0:
        return np.empty(0, dtype=np.int32)
    selected: list[int] = []
    remaining = np.ones(count, dtype=bool)
    for _ in range(count):
        candidates = np.flatnonzero(remaining)
        utility = unary_logits[candidates].copy()
        if selected:
            compatibility = 1.0 / (1.0 + np.exp(-np.clip(pair_logits[np.ix_(candidates, selected)], -30, 30)))
            utility += pair_selection_weight * compatibility.max(axis=1)
        winner = int(candidates[int(np.argmax(utility))])
        selected.append(winner)
        remaining[winner] = False
    return np.asarray(selected, dtype=np.int32)


@torch.no_grad()
def evaluate_reranker(model: GraphAwareListwiseReranker, examples: Sequence[CandidateExample],
                      config: RerankerConfig) -> dict[str, float | int]:
    eligible = [e for e in examples if e.total_gold_count > 0]
    if not eligible:
        return {"queries": 0, "recall_at_k": 0.0, "complete_at_k": 0.0, "at_least_one_at_k": 0.0,
                "candidate_complete_recall": 0.0}
    total_found = total_gold = complete = at_least_one = candidate_complete = 0
    model.eval()
    for start in range(0, len(eligible), config.batch_size):
        chunk = eligible[start:start + config.batch_size]
        for example, (order, _, _) in zip(chunk, model.score_examples(chunk, config.device)):
            k = min(config.retrieval_top_k, len(order))
            found = int(example.gold_mask[order[:k]].sum())
            total_found += found
            total_gold += example.total_gold_count
            complete += int(found == example.total_gold_count)
            at_least_one += int(found > 0)
            candidate_complete += int(example.visible_candidate_complete)
    n = len(eligible)
    return {"queries": n, "recall_at_k": total_found / max(total_gold, 1), "complete_at_k": complete / n,
            "at_least_one_at_k": at_least_one / n, "candidate_complete_recall": candidate_complete / n}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_reranker(model: GraphAwareListwiseReranker, train_examples: Sequence[CandidateExample],
                   valid_examples: Sequence[CandidateExample], config: RerankerConfig,
                   checkpoint_path: Optional[str] = None) -> list[dict[str, float | int]]:
    """Train with early stopping on validation Complete@k; the best epoch's weights are restored."""
    config.validate()
    train_examples = [e for e in train_examples if e.gold_mask.any()]
    if not train_examples:
        raise ValueError("no training examples contain resolved gold passages; check the gold resolver")
    seed_everything(config.seed)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_complete, best_state, stale_epochs = -1.0, None, 0
    history: list[dict[str, float | int]] = []
    indices = np.arange(len(train_examples))
    for epoch in range(1, config.epochs + 1):
        np.random.shuffle(indices)
        model.train()
        running_loss, steps = 0.0, 0
        running_parts = {"listwise": 0.0, "binary": 0.0, "complete_margin": 0.0, "pair": 0.0}
        for start in range(0, len(indices), config.batch_size):
            batch = collate_examples([train_examples[int(i)] for i in indices[start:start + config.batch_size]],
                                     device)
            optimizer.zero_grad(set_to_none=True)
            unary, pair = model(batch["query"], batch["passages"], batch["scalar"], batch["pair"], batch["mask"])
            loss, parts = reranker_loss(unary, pair, batch["gold"], batch["mask"], config)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            for name, value in parts.items():
                running_parts[name] += value
            steps += 1
        validation = evaluate_reranker(model, valid_examples, config)
        history.append({"epoch": epoch, "train_loss": running_loss / max(steps, 1),
                        **{f"train_{k}": v / max(steps, 1) for k, v in running_parts.items()},
                        **{f"valid_{k}": v for k, v in validation.items()}})
        logger.info("epoch=%d loss=%.4f valid_Complete@%d=%.4f", epoch, history[-1]["train_loss"],
                    config.retrieval_top_k, validation["complete_at_k"])
        if float(validation["complete_at_k"]) > best_complete + 1e-8:
            best_complete = float(validation["complete_at_k"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
            if checkpoint_path:
                save_checkpoint(checkpoint_path, model, config, history)
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path:
        save_checkpoint(checkpoint_path, model, config, history)
    return history


def save_checkpoint(path: str, model: GraphAwareListwiseReranker, config: RerankerConfig,
                    history: Optional[list[dict[str, Any]]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(config), "embedding_dim": model.embedding_dim,
                "scalar_feature_names": SCALAR_FEATURE_NAMES, "pair_feature_names": PAIR_FEATURE_NAMES,
                "history": history or []}, path)


def load_checkpoint(path: str, device: Optional[str] = None
                    ) -> tuple[GraphAwareListwiseReranker, RerankerConfig, list[dict[str, Any]]]:
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=target, weights_only=False)
    if tuple(payload["scalar_feature_names"]) != SCALAR_FEATURE_NAMES or \
            tuple(payload["pair_feature_names"]) != PAIR_FEATURE_NAMES:
        raise ValueError("checkpoint feature schema is incompatible")
    config = RerankerConfig(**payload["config"])
    config.device = target
    model = GraphAwareListwiseReranker(int(payload["embedding_dim"]), config)
    model.load_state_dict(payload["state_dict"])
    model.to(target)
    return model, config, list(payload.get("history", []))


class AdaptiveReranker:
    """The full SpearRAG retriever: stages 1-2 + fusion, then the trained listwise reranker."""

    def __init__(self, retriever: SpearRAGRetriever, model: GraphAwareListwiseReranker,
                 config: RerankerConfig) -> None:
        self.retriever = retriever
        self.model = model
        self.config = config
        self.extractor = FeatureExtractor(retriever, config)

    def retrieve(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        results = []
        for row, query_embedding in zip(rows, self.extractor.encode_queries(rows)):
            example = self.extractor.build(row, query_embedding, gold_indices=None, training=False)
            order, unary, pair = self.model.score_examples([example], self.config.device)[0]
            chosen_local = order[:min(self.config.retrieval_top_k, len(order))]
            chosen = example.passage_indices[chosen_local]
            texts = [self.retriever._passage_texts[int(i)] for i in chosen]
            results.append({
                "question": str(row["question"]),
                "gold_answer": row.get("answer", ""),
                "sorted_passage": texts,
                "sorted_passage_scores": [float(unary[int(local)]) for local in chosen_local],
                "ordered_passage": texts,   # the reranker's greedy order is also the generator's order
                "diagnostics": {"full_candidate_count": example.full_candidate_count,
                                "reranker_candidate_count": len(example.passage_indices),
                                "selected_passage_indices": [int(i) for i in chosen],
                                **example.metadata},
            })
        return results
