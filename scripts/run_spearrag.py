"""Run SpearRAG end to end on one dataset: index -> train the reranker -> retrieve -> generate -> judge.

Protocol (the paper's):
  * the reranker is trained on a separate TRAIN corpus/question set (``{dataset}_regimetrain.json``; 100 of its
    questions are held out for early stopping) and applied, frozen, to the TEST corpus;
  * test corpus = the LinearRAG/EHRAG release (``{dataset}_linearrag_regime.json``, ~800-word lowercased chunks);
  * generation: gpt-4o-mini, temperature 0, CoT prompt ("Thought: ... Answer: ..."); judge: gpt-4o-mini 0/1;
  * Accuracy = (Contain-Acc + GPT-Acc) / 2 (Medical: GPT-Acc only).
Indexing and retrieval never call an LLM; ``--skip-generation`` therefore costs $0 (timing / R@5 runs).

Examples
  python scripts/run_spearrag.py --dataset 2wiki --seeds 42,43
  python scripts/run_spearrag.py --dataset hotpotqa --ablation no_subgraph_ppr --seeds 42
  python scripts/run_spearrag.py --dataset 2wiki --test-data-file 2wiki_linearrag_regime_gold.json \
         --skip-generation --ppr-scope global                    # R@5 / Complete@5 only, $0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ABLATIONS = {
    # name                       : what is removed                                    (research-code arm)
    "none": "full SpearRAG",
    "no_bridge": "Bridge Propagation (seeds only, max_bridge_hops=0)",               # no_bridge
    "no_candidate_generation": "Bridged Candidate Generation (dense top-128 only)",  # dense_cand
    "no_subgraph_ppr": "Subgraph PPR (restart distribution only)",                   # no_ppr_l2_diffusion
    "no_graph_features": "graph-derived scalar features of the reranker",            # no_graph_feats_only
    "no_rerank": "Adaptive Reranking (fused score + chain order)",                   # baseline_rank
}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", required=True, choices=["2wiki", "hotpotqa", "musique", "medical"])
p.add_argument("--data-dir", default=str(ROOT / "data"))
p.add_argument("--train-data-file", default=None, help="default: {dataset}_regimetrain.json (Medical: 2wiki_regimetrain.json)")
p.add_argument("--test-data-file", default=None, help="default: {dataset}_linearrag_regime.json")
p.add_argument("--spacy-model", default=None, help="default: en_core_web_trf (Medical: en_core_sci_scibert)")
p.add_argument("--seeds", default="42", help="reranker seeds, e.g. 42,43")
p.add_argument("--ablation", default="none", choices=list(ABLATIONS))
p.add_argument("--ppr-scope", default="subgraph", choices=["subgraph", "global"],
               help="global = same restart on the full passage+entity graph (control experiment)")
p.add_argument("--overrides", default="{}", help='JSON of SpearRAGConfig fields, e.g. \'{"ppr_damping": 0.7}\'')
p.add_argument("--top-k", type=int, default=5)
p.add_argument("--candidate-limit", type=int, default=128)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--train-valid-size", type=int, default=100)
p.add_argument("--n-test-queries", type=int, default=None, help="truncate the test questions (smoke tests)")
p.add_argument("--skip-generation", action="store_true", help="retrieval only: no LLM call, answer metrics invalid")
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--working-dir", default="/tmp/spearrag_working")
p.add_argument("--out-dir", default=str(ROOT / "results"))
p.add_argument("--llm-model", default="gpt-4o-mini")
p.add_argument("--max-tokens", type=int, default=512)
p.add_argument("--n-bootstrap", type=int, default=2000)
args = p.parse_args()

if not args.skip_generation and not os.environ.get("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY (or pass --skip-generation for a $0 retrieval-only run).")
os.environ.setdefault("SPEARRAG_SPACY_GPU", str(args.gpu))   # spaCy (cupy) binds its GPU separately from torch

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from spearrag import token_meter  # noqa: E402
from spearrag.config import ProjectConfig  # noqa: E402
from spearrag.reranker import (AdaptiveReranker, FeatureExtractor, RerankerConfig, build_examples,  # noqa: E402
                               create_reranker, doc_index_gold_resolver, seed_everything, train_reranker)
from spearrag.retriever import SpearRAGConfig, SpearRAGRetriever  # noqa: E402
from spearrag.utils import normalize_answer  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
SEEDS = [int(s) for s in args.seeds.split(",") if s.strip()]
train_file = args.train_data_file or ("2wiki_regimetrain.json" if args.dataset == "medical"
                                      else f"{args.dataset}_regimetrain.json")
test_file = args.test_data_file or f"{args.dataset}_linearrag_regime.json"
spacy_model = args.spacy_model or ("en_core_sci_scibert" if args.dataset == "medical" else "en_core_web_trf")
Path(args.out_dir).mkdir(parents=True, exist_ok=True)

train_raw = json.load(open(Path(args.data_dir) / train_file))
test_raw = json.load(open(Path(args.data_dir) / test_file))
train_documents, train_queries = train_raw["documents"], train_raw["queries"]
test_documents, test_queries = test_raw["documents"], test_raw["queries"]
if args.n_test_queries:
    test_queries = test_queries[:args.n_test_queries]
# The test release re-samples the same question pool as the training sample: shared questions are dropped
# from TRAINING (never from the test set).
overlap = {q["question"] for q in train_queries} & {q["question"] for q in test_queries}
train_queries = [q for q in train_queries if q["question"] not in overlap]
valid_rows, train_rows = train_queries[:args.train_valid_size], train_queries[args.train_valid_size:]
print(f"dataset={args.dataset} ablation={args.ablation} ppr_scope={args.ppr_scope} seeds={SEEDS}\n"
      f"  train: {train_file} ({len(train_documents)} docs, {len(train_rows)} train / {len(valid_rows)} valid questions;"
      f" {len(overlap)} dropped as test overlap)\n  test : {test_file} ({len(test_documents)} docs, "
      f"{len(test_queries)} questions)", flush=True)

train_doc_index = {d["id"]: i for i, d in enumerate(train_documents)}
test_doc_index = {d["id"]: i for i, d in enumerate(test_documents)}
test_gold = [{test_doc_index[d] for d in q.get("gold_doc_ids", []) if d in test_doc_index} for q in test_queries]

emb_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)


def retriever_config() -> SpearRAGConfig:
    cfg = SpearRAGConfig(retrieval_top_k=args.top_k, ppr_scope=args.ppr_scope)
    if args.ablation == "no_bridge":
        cfg.max_bridge_hops = 0
    elif args.ablation == "no_candidate_generation":
        cfg.enable_bridge_candidates = False
        cfg.passage_neighbor_expansion = 0
    elif args.ablation == "no_subgraph_ppr":
        cfg.enable_subgraph_ppr = False
    for k, v in json.loads(args.overrides).items():
        assert hasattr(cfg, k), f"unknown SpearRAGConfig field: {k}"
        setattr(cfg, k, v)
    return cfg


def build_index(name: str, documents) -> tuple[SpearRAGRetriever, float]:
    r = SpearRAGRetriever(ProjectConfig(dataset_name=f"{args.dataset}_{name}", embedding_model=emb_model,
                                        working_dir=str(Path(args.working_dir) / name), spacy_model=spacy_model),
                          retriever_config())
    t0 = time.perf_counter()
    r.index([f"{i}:{d['text']}" for i, d in enumerate(documents)])
    return r, time.perf_counter() - t0


token_meter.install()   # proves indexing / retrieval make no LLM call (method_only = 0)
token_meter.set_phase("index")
train_retriever, train_idx_time = build_index("train", train_documents)
print(f"Train index done in {train_idx_time:.1f}s", flush=True)
test_retriever, test_idx_time = build_index("test", test_documents)
print(f"Test index done in {test_idx_time:.1f}s", flush=True)

# ---------------------------------------------------------------- generation and judging (unchanged from the paper runs)
PROMPT_SYSTEM = (
    "As an advanced reading comprehension assistant, your task is to analyze text "
    "passages and corresponding questions meticulously. Your response start after "
    '"Thought: ", where you will methodically break down the reasoning process, '
    'illustrating how you arrive at conclusions. Conclude with "Answer: " to present '
    "a concise, definitive response, devoid of additional elaborations."
)
JUDGE_SYSTEM = (
    "You are an evaluation assistant. "
    "Given a question, a gold answer, and a predicted answer, "
    "reply with 1 if the predicted answer is correct, else 0. "
    "Reply with only the number."
)


def build_prompt(passages, question):
    ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return [{"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}\nThought: "}]


def generate_answers(msgs):
    from openai import OpenAI
    client = OpenAI()

    def one(msg):
        try:
            r = client.chat.completions.create(model=args.llm_model, messages=msg, temperature=0,
                                               max_tokens=args.max_tokens)
            text = r.choices[0].message.content or ""
            return text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()
        except Exception:
            return ""   # counted as wrong; a run with many empty answers indicates an API problem
    with ThreadPoolExecutor(max_workers=32) as ex:
        return list(ex.map(one, msgs))


def judge(preds, golds, questions):
    from openai import OpenAI
    client = OpenAI()

    def one(item):
        pred, gold, q = item
        try:
            r = client.chat.completions.create(
                model=args.llm_model, temperature=0, max_tokens=4,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": f"Question: {q}\nGold: {gold}\nPredicted: {pred}"}])
            return int((r.choices[0].message.content or "0").strip())
        except Exception:
            return 0
    with ThreadPoolExecutor(max_workers=32) as ex:
        return list(ex.map(one, zip(preds, golds, questions)))


def em(p, g): return float(normalize_answer(p) == normalize_answer(g))
def contain(p, g): return float(normalize_answer(g) in normalize_answer(p))


def f1(pred, gold):
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = sum(min(pt.count(t), gt.count(t)) for t in set(pt) & set(gt))
    if common == 0:
        return 0.0
    prec, rec = common / len(pt), common / len(gt)
    return 2 * prec * rec / (prec + rec)


def passage_index(text):
    try:
        return int(text.split(":", 1)[0])
    except Exception:
        return None


# ---------------------------------------------------------------- per seed: train reranker, retrieve, answer
golds = [q["answer"] for q in test_queries]
questions = [q["question"] for q in test_queries]
per_seed, per_query = [], {k: [] for k in ("em", "f1", "c_acc", "gpt_acc", "r5", "complete5")}
first_contexts = first_preds = None
for seed in SEEDS:
    t_seed = time.perf_counter()
    rcfg = RerankerConfig(candidate_limit=args.candidate_limit, retrieval_top_k=args.top_k, epochs=args.epochs,
                          device=DEVICE, seed=seed,
                          feature_ablation="minus_graph_scalars" if args.ablation == "no_graph_features" else "full")
    extractor = FeatureExtractor(train_retriever, rcfg)
    resolver = doc_index_gold_resolver(train_doc_index)
    train_examples, _ = build_examples(extractor, train_rows, training=True, gold_resolver=resolver)
    valid_examples, _ = build_examples(extractor, valid_rows, training=False, gold_resolver=resolver)
    seed_everything(seed)   # seeds the model's initial weights too (see README, reproducibility notes)
    model = create_reranker(train_retriever, rcfg)
    history = train_reranker(model, train_examples, valid_examples, rcfg)
    best_valid = max((h["valid_complete_at_k"] for h in history), default=0.0)
    print(f"[seed={seed}] reranker: {len(history)} epochs, best valid Complete@{args.top_k}={best_valid:.4f}", flush=True)

    token_meter.set_phase("retrieval")
    system = test_retriever if args.ablation == "no_rerank" else AdaptiveReranker(test_retriever, model, rcfg)
    t0 = time.perf_counter()
    results = system.retrieve(test_queries)
    ret_time = time.perf_counter() - t0
    print(f"[seed={seed}] retrieval {ret_time:.1f}s ({ret_time / max(len(test_queries), 1):.3f}s/query)", flush=True)

    r5 = np.zeros(len(test_queries))
    complete5 = np.zeros(len(test_queries))
    for qi, gold in enumerate(test_gold):
        if gold:
            top = {passage_index(t) for t in results[qi]["sorted_passage"][:args.top_k]} - {None}
            r5[qi] = len(gold & top) / len(gold)
            complete5[qi] = float(gold <= top)

    if args.skip_generation:
        preds = [""] * len(test_queries)
        scores = {k: np.zeros(len(test_queries)) for k in ("em", "f1", "c_acc", "gpt_acc")}
    else:
        token_meter.set_phase("generation")
        preds = generate_answers([build_prompt(results[i]["ordered_passage"], questions[i])
                                  for i in range(len(test_queries))])
        token_meter.set_phase("judge")
        scores = {"em": np.array([em(p, g) for p, g in zip(preds, golds)]),
                  "f1": np.array([f1(p, g) for p, g in zip(preds, golds)]),
                  "c_acc": np.array([contain(p, g) for p, g in zip(preds, golds)]),
                  "gpt_acc": np.array(judge(preds, golds, questions), dtype=np.float64)}
        empty = sum(1 for x in preds if not x)
        if empty > 0.02 * len(preds):
            print(f"WARNING: {empty} empty answers -- check the API key / credit", flush=True)
    scores.update(r5=r5, complete5=complete5)
    for k in per_query:
        per_query[k].append(scores[k])
    if first_contexts is None:
        first_contexts = [list(r["ordered_passage"][:args.top_k]) for r in results]
        first_preds = list(preds)
    summary = {"seed": seed, "epochs_trained": len(history), "best_valid_complete_at_k": best_valid,
               **{k: round(float(v.mean()), 4) for k, v in scores.items()},
               "time_s": round(time.perf_counter() - t_seed, 1), "ret_time_s": round(ret_time, 1)}
    if args.dataset != "medical":
        summary["accuracy"] = round((summary["c_acc"] + summary["gpt_acc"]) / 2, 4)
    per_seed.append(summary)
    print(f"[seed={seed}] " + "  ".join(f"{k}={summary[k]}" for k in
          ("r5", "complete5", "em", "f1", "c_acc", "gpt_acc", "accuracy") if k in summary), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean_std(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std(ddof=0)), 4)}


def bootstrap(per_q, n, seed=0):
    rng = np.random.default_rng(seed)
    means = [per_q[rng.integers(0, len(per_q), len(per_q))].mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(per_q.mean()), 4), "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4)}


seed_avg = {k: np.mean(np.stack(v), axis=0) for k, v in per_query.items()}
aggregate = {"across_seeds_mean_std": {k: mean_std([s[k] for s in per_seed]) for k in per_query},
             "seed_averaged_bootstrap_ci95": {k: bootstrap(v, args.n_bootstrap) for k, v in seed_avg.items()}}
if args.dataset != "medical":
    aggregate["across_seeds_mean_std"]["accuracy"] = mean_std([s["accuracy"] for s in per_seed])

out = {
    "method": "SpearRAG", "dataset": args.dataset, "train_data_file": train_file, "test_data_file": test_file,
    "spacy_model": spacy_model, "ablation": args.ablation, "ppr_scope": args.ppr_scope,
    "overrides": json.loads(args.overrides), "llm_model": args.llm_model, "retrieval_only": bool(args.skip_generation),
    "seeds": SEEDS, "test_size": len(test_queries), "n_test_documents": len(test_documents),
    "train_idx_time_s": round(train_idx_time, 1), "test_idx_time_s": round(test_idx_time, 1),
    "per_seed": per_seed, "aggregate": aggregate, "token_usage": token_meter.report(),
    "retrieved_contexts": first_contexts, "predictions": first_preds,
    "query_ids": [q.get("id", str(i)) for i, q in enumerate(test_queries)],
    "question_types": [q.get("question_type", "") for q in test_queries],
    "questions": questions, "gold_answers": golds,
    "seed_avg_gpt_acc_per_query": seed_avg["gpt_acc"].round(4).tolist(),
    "seed_avg_em_per_query": seed_avg["em"].round(4).tolist(),
}
suffix = ("" if args.ablation == "none" else f"_{args.ablation}") + ("_global_ppr" if args.ppr_scope == "global" else "")
out_path = Path(args.out_dir) / f"{args.dataset}_spearrag{suffix}.json"
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Saved -> {out_path}")
