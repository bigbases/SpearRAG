"""SpearRAG: Subgraph Propagation with Entity-bridged Adaptive Reranking Retrieval-Augmented Generation.

    from spearrag import ProjectConfig, SpearRAGConfig, SpearRAGRetriever        # stages 1-2 + fusion
    from spearrag.reranker import RerankerConfig, AdaptiveReranker               # stage 3

Heavy dependencies (torch, spaCy, sentence-transformers) load only when a name is used.
"""
from importlib import import_module

_EXPORTS = {
    "ProjectConfig": ("spearrag.config", "ProjectConfig"),
    "SpearRAGConfig": ("spearrag.retriever", "SpearRAGConfig"),
    "SpearRAGRetriever": ("spearrag.retriever", "SpearRAGRetriever"),
    "RerankerConfig": ("spearrag.reranker", "RerankerConfig"),
    "AdaptiveReranker": ("spearrag.reranker", "AdaptiveReranker"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'spearrag' has no attribute {name!r}")
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module), attr)
    globals()[name] = value
    return value
