"""Cold-start indexing time of SpearRAG on one corpus, with the LLM token meter on ($0).

    python scripts/with_spacy_batch.py 128 scripts/measure_indexing.py --corpus data/scale_5m.json
Delete --working-dir first for a true cold start (NER / embedding caches live there).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True, help="JSON with a 'documents' list of {'id','text'}")
ap.add_argument("--spacy-model", default="en_core_web_trf")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--working-dir", default="/tmp/spearrag_index_timing")
ap.add_argument("--out", default=None)
args = ap.parse_args()
os.environ.setdefault("SPEARRAG_SPACY_GPU", str(args.gpu))
shutil.rmtree(args.working_dir, ignore_errors=True)

from sentence_transformers import SentenceTransformer  # noqa: E402
from spearrag import token_meter  # noqa: E402
from spearrag.config import ProjectConfig  # noqa: E402
from spearrag.retriever import SpearRAGRetriever  # noqa: E402

docs = json.load(open(args.corpus))["documents"]
emb = SentenceTransformer("all-mpnet-base-v2", device=f"cuda:{args.gpu}")
token_meter.install()
token_meter.set_phase("index")
retriever = SpearRAGRetriever(ProjectConfig(dataset_name="index_timing", embedding_model=emb,
                                            working_dir=args.working_dir, spacy_model=args.spacy_model))
t0 = time.perf_counter()
retriever.index([f"{i}:{d['text']}" for i, d in enumerate(docs)])
elapsed = time.perf_counter() - t0
usage = token_meter.report()["method_only"]
result = {"system": "SpearRAG", "corpus": args.corpus, "n_chunks": len(docs), "indexing_time_s": round(elapsed, 1),
          "index_prompt_tokens": usage["prompt"], "index_completion_tokens": usage["completion"],
          "index_llm_calls": usage["calls"], "spacy_model": args.spacy_model, "embedding_model": "all-mpnet-base-v2"}
print(json.dumps(result, indent=1))
if args.out:
    Path(args.out).write_text(json.dumps(result, indent=1))
shutil.rmtree(args.working_dir, ignore_errors=True)
