# SpearRAG: Subgraph Propagation with Entity-bridged Adaptive Reranking for Retrieval-Augmented Generation

> Graph-based retrieval-augmented generation (RAG) links evidence across documents for multi-hop question answering but is costly at scale. Knowledge graphs built with large language models (LLMs) cost tokens in proportion to corpus size; LLM-free systems avoid this cost but rely on propagation over the whole graph, typically Personalized PageRank (PPR), a random walk that restarts from the query’s entities, both to reach distant evidence and to rank it, so per-query latency grows with the corpus. We present SpearRAG (Subgraph Propagation with Entity-bridged Adaptive Reranking for RAG), an LLM-free graph-based RAG framework that decouples corpus-wide evidence discovery from graph propagation. Over an entity-shared passage graph built in linear time, entity-bridged candidate generation combines dense retrieval with multi-hop entity expansion to collect a query-induced subgraph; subgraph propagation runs the same PPR only there, and adaptive reranking with a learned listwise reranker selects the evidence. As distractors grow a 2Wiki corpus from 0.75M to 40.75M tokens, the per-query latency of SpearRAG grows only 2.1×, versus 22.7× for LinearRAG, the strongest baseline, so its speedup widens from 1.4× to 14.5×; on the 350M-token HotpotQA Wikipedia corpus, it is 47× faster (0.87 s vs. 40.8 s) and retrieves more gold evidence at comparable answer accuracy. On four benchmarks, SpearRAG is also the most accurate system on 2WikiMultiHopQA and Medical, the most accurate LLM-free system on MuSiQue, and within 0.6 points of the best on HotpotQA, with a significant macro-averaged gain, and indexes as fast as the other LLM-free systems with zero LLM tokens.

## Layout

```
SpearRAG/
├── spearrag/
│   ├── retriever.py      SpearRAGConfig, SpearRAGRetriever  (offline index, stages ① ② and fusion)
│   ├── reranker.py       RerankerConfig, model, training, AdaptiveReranker  (stage ③, the full method)
│   ├── config.py         ProjectConfig  (embedding model, spaCy model, cache dir, batching)
│   ├── ner.py            spaCy NER (GPU when cupy is available)
│   ├── embedding_store.py   parquet-backed embedding cache
│   ├── token_meter.py    counts every OpenAI call (proves indexing/retrieval use 0 LLM tokens)
│   └── utils.py
├── scripts/
│   ├── run_spearrag.py       index → train reranker → retrieve → generate → judge  (main tables, ablations)
│   ├── paired_test.py        per-question paired significance of ablations
│   ├── measure_indexing.py   cold-start indexing time (scalability)
│   ├── with_spacy_batch.py   fixed spaCy batch for large corpora
│   └── check_equivalence.py  proof that this package matches the research code
├── docs/RESULTS.md       final paper numbers
├── data/                datasets
└── requirements.txt
```

## Install

Python 3.9, CUDA 11.8 (tested on RTX A4000 16 GB). `pip install -r requirements.txt`, then
`python -m spacy download en_core_web_trf` (Medical also needs `en_core_sci_scibert` from scispaCy) and
`pip install cupy-cuda11x` so spaCy's transformer NER runs on the GPU (≈15x faster than CPU).

## Run

The paper uses at most 20 PPR iterations ($T_{\max}$ = 20, Table 1 of the paper), while `SpearRAGConfig` defaults to
`ppr_max_iterations = 80`. Pass `--overrides '{"ppr_max_iterations": 20}'` (as below) to reproduce the paper's setting.

```bash
cd SpearRAG
PY=python
export OPENAI_API_KEY=...          # only needed for generation/judging
PAPER='{"ppr_max_iterations": 20}'  # paper setting (T_max = 20)

# main table (one row): ~10 min per seed per dataset, ~$0.95 per seed (Medical ~$2)
$PY scripts/run_spearrag.py --dataset 2wiki --seeds 42,43,44 --overrides "$PAPER"

# ablations: no_bridge | no_candidate_generation | no_subgraph_ppr | no_graph_features | no_rerank
$PY scripts/run_spearrag.py --dataset hotpotqa --ablation no_subgraph_ppr --overrides "$PAPER"
$PY scripts/paired_test.py --ablations no_bridge no_candidate_generation no_subgraph_ppr no_graph_features no_rerank

# retrieval only ($0): R@5 / Complete@5 and timing; --ppr-scope global is the subgraph-vs-global control
$PY scripts/run_spearrag.py --dataset 2wiki --test-data-file 2wiki_linearrag_regime_gold.json --skip-generation --overrides "$PAPER"
$PY scripts/run_spearrag.py --dataset 2wiki --test-data-file 2wiki_linearrag_regime_gold.json --skip-generation --ppr-scope global --overrides "$PAPER"

# large corpora: fix the spaCy batch (a 16 GB GPU runs out of memory otherwise)
$PY scripts/with_spacy_batch.py 128 scripts/run_spearrag.py --dataset 2wiki --test-data-file haystack_2wiki_20m.json --skip-generation --overrides "$PAPER"
$PY scripts/with_spacy_batch.py 128 scripts/measure_indexing.py --corpus data/scale_5m.json
```

Results go to `results/{dataset}_spearrag[_{ablation}][_global_ppr].json` (per-seed metrics, bootstrap CIs,
retrieved contexts, answers, per-question judge scores, OpenAI token usage, index/retrieval times).
For timing, run one job at a time on an otherwise idle machine and delete `--working-dir` first (caches).

Library use:

```python
from sentence_transformers import SentenceTransformer
from spearrag import ProjectConfig, SpearRAGRetriever, SpearRAGConfig
from spearrag.reranker import (RerankerConfig, FeatureExtractor, AdaptiveReranker, build_examples,
                               create_reranker, train_reranker, doc_index_gold_resolver, seed_everything)

emb = SentenceTransformer("all-mpnet-base-v2")
rcfg = SpearRAGConfig(ppr_max_iterations=20)   # paper setting (T_max = 20)
train_r = SpearRAGRetriever(ProjectConfig("train", emb), rcfg); train_r.index(train_passages)
test_r = SpearRAGRetriever(ProjectConfig("test", emb), rcfg);   test_r.index(test_passages)
cfg = RerankerConfig()
ext = FeatureExtractor(train_r, cfg)
resolve = doc_index_gold_resolver({doc_id: i for i, doc_id in enumerate(train_doc_ids)})
tr, _ = build_examples(ext, train_rows, training=True, gold_resolver=resolve)
va, _ = build_examples(ext, valid_rows, training=False, gold_resolver=resolve)
seed_everything(cfg.seed); model = create_reranker(train_r, cfg); train_reranker(model, tr, va, cfg)
top5 = AdaptiveReranker(test_r, model, cfg).retrieve([{"question": "..."}])[0]["sorted_passage"]
```
Passages are indexed as `"{i}:{text}"`; the leading index is how gold ids and R@5 are resolved.


## Pairwise selection term

With the paper's listwise-only objective, the reranker's pairwise compatibility head (`pair_query`, `pair_key`,
`pair_edge`) receives no gradient, yet greedy selection adds `0.35 × sigmoid(pair logit)` after the first pick
(η = 0.35, Eq. (17) of the paper), so that term comes from the head's initial weights. As reported in the paper
(Section 4.5), setting `pair_selection_weight = 0` changes the top-5 for only 1.3–3.1% of questions and accuracy by at
most 0.70 points; candidate interactions are captured mainly by the listwise attention. Training the head
(`pair_loss_weight > 0`) is untested.
