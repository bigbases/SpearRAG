"""Check that this package computes exactly what the research code computed for the paper ($0, no LLM).

Indexes one corpus with both implementations and compares, for every question:
  * stages 1-2 + fusion: candidate set, Subgraph PPR, entity coverage, fused score
  * the fused ranking returned by retrieve() (top-k and generator order)
  * the reranker's input features (the research code's three community columns were constant zeros under
    ``--no-community``; they are dropped here, every other column must match)

Research code: ``QACFRetriever`` / ``QACFFeatureExtractor`` in ``--research-root`` (the handover folder),
configured exactly as ``final_test_eval.py --no-community``.

    python scripts/check_equivalence.py --research-root /workspace/GraphRAG/handover --dataset 2wiki
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--research-root", required=True)
ap.add_argument("--dataset", default="2wiki")
ap.add_argument("--test-data-file", default=None)
ap.add_argument("--n-queries", type=int, default=None)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--work", default="/tmp/spearrag_equivalence")
args = ap.parse_args()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, args.research_root)
os.environ.setdefault("SPEARRAG_SPACY_GPU", str(args.gpu))
os.environ.setdefault("QACF_SPACY_GPU", str(args.gpu))

from sentence_transformers import SentenceTransformer  # noqa: E402
from commurag import CommuRAGConfig  # noqa: E402
from commurag.qacf_rag import QACFRetriever, QACFConfig  # noqa: E402
from commurag.qacf_listwise_reranker import GraphListwiseConfig, QACFFeatureExtractor  # noqa: E402
from spearrag.config import ProjectConfig  # noqa: E402
from spearrag.reranker import FeatureExtractor, RerankerConfig  # noqa: E402
from spearrag.retriever import SpearRAGConfig, SpearRAGRetriever  # noqa: E402

spacy_model = "en_core_sci_scibert" if args.dataset == "medical" else "en_core_web_trf"
data = json.load(open(ROOT / "data" / (args.test_data_file or f"{args.dataset}_linearrag_regime.json")))
docs = data["documents"]
queries = data["queries"][:args.n_queries] if args.n_queries else data["queries"]
passages = [f"{i}:{d['text']}" for i, d in enumerate(docs)]
emb = SentenceTransformer("all-mpnet-base-v2", device=f"cuda:{args.gpu}")

shutil.rmtree(args.work, ignore_errors=True)
old = QACFRetriever(
    CommuRAGConfig(dataset_name="eq", embedding_model=emb, llm_model=None, working_dir=f"{args.work}/old",
                   retrieval_top_k=5, spacy_model=spacy_model),
    QACFConfig(retrieval_top_k=5, enable_community_candidates=False, final_route_weight=0.0,
               enable_query_gated_edges=False, enable_community_flow=False, build_communities=False))
old.index(passages)
# Reuse the NER / embedding caches (same corpus fingerprint) so both indexes start from identical inputs.
shutil.copytree(f"{args.work}/old/eq/qacf", f"{args.work}/new/eq/spearrag")
new = SpearRAGRetriever(ProjectConfig(dataset_name="eq", embedding_model=emb, working_dir=f"{args.work}/new",
                                      spacy_model=spacy_model), SpearRAGConfig(retrieval_top_k=5))
new.index(passages)

report = {"dataset": args.dataset, "queries": len(queries)}
report["passage_graph_identical"] = bool((old._passage_graph != new._passage_graph).nnz == 0)
report["contain_identical"] = bool((old._contain != new._contain).nnz == 0)

ox = QACFFeatureExtractor(old, GraphListwiseConfig(candidate_limit=128, retrieval_top_k=5,
                                                   community_ablation="minus_reranker_community_features"))
nx = FeatureExtractor(new, RerankerConfig(candidate_limit=128, retrieval_top_k=5))
q_emb = nx.encode_queries(queries)
OLD_SCALAR_KEEP = [i for i in range(13) if i not in (5, 12)]   # drop community_route_z, inverse_community_size
OLD_PAIR_KEEP = [0, 1, 3, 4]                                    # drop same_community
mismatch = {"candidates": 0, "scalar_features": 0, "pair_features": 0, "fused_scores": 0,
            "community_columns_nonzero": 0}
max_diff = 0.0
for q, e in zip(queries, q_emb):
    a = ox.build(q, e, gold_indices=None, training=False)
    b = nx.build(q, e, gold_indices=None, training=False)
    if not np.array_equal(a.passage_indices, b.passage_indices) or a.full_candidate_count != b.full_candidate_count:
        mismatch["candidates"] += 1
        continue
    if np.any(a.scalar_features[:, [5, 12]]) or np.any(a.pair_features[..., 2]):
        mismatch["community_columns_nonzero"] += 1
    for name, x, y in (("scalar_features", a.scalar_features[:, OLD_SCALAR_KEEP], b.scalar_features),
                       ("pair_features", a.pair_features[..., OLD_PAIR_KEEP], b.pair_features),
                       ("fused_scores", a.baseline_scores, b.baseline_scores)):
        d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
        max_diff = max(max_diff, d)
        mismatch[name] += int(d > 0.0)
report["reranker_input_mismatches"] = mismatch
report["max_abs_difference"] = max_diff

ro, rn = old.retrieve(queries), new.retrieve(queries)
report["fused_ranking_mismatches"] = {
    "sorted_passage": sum(x["sorted_passage"] != y["sorted_passage"] for x, y in zip(ro, rn)),
    "ordered_passage": sum(x["ordered_passage"] != y["ordered_passage"] for x, y in zip(ro, rn)),
}
report["identical"] = (report["passage_graph_identical"] and report["contain_identical"] and max_diff == 0.0
                       and not any(mismatch.values()) and not any(report["fused_ranking_mismatches"].values()))
print(json.dumps(report, indent=1))
out = ROOT / "results" / f"equivalence_{args.dataset}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=1))
shutil.rmtree(args.work, ignore_errors=True)
