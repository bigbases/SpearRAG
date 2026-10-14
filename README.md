# SpearRAG

**SpearRAG: Subgraph Propagation with Entity-bridged Adaptive Reranking Retrieval-Augmented Generation**

An LLM-free GraphRAG retriever for multi-hop QA. Indexing uses spaCy NER and sentence embeddings only (0 LLM
tokens). At query time, global recall comes from entity-bridged candidate generation, and graph propagation
runs only on the query-induced subgraph, so retrieval cost is bounded by the subgraph instead of the corpus
(HotpotQA fullwiki, ~350M tokens: 1.35 s/query vs LinearRAG 40.8 s, with higher R@5).
Final numbers: [`docs/RESULTS.md`](docs/RESULTS.md).

## Method → code

```
[Offline]  Entity-Shared Passage Graph Construction
[Online]   ① Entity Bridging (E)  →  ② Subgraph Propagation (SP)  →  ③ Adaptive Reranking (AR)
```

| Module | Component | What it does | Code |
|---|---|---|---|
| Offline | Entity Extraction | spaCy NER; entity / sentence / passage stores and incidence matrices | `SpearRAGRetriever.index` |
| | Entity-Shared Passage Graph | passages linked by the cosine of their entity TF-IDF vectors, top-24 neighbours, built block-wise (memory linear in corpus size) | `_build_passage_graph` |
| ① E | Entity Seeding | question mentions → indexed entities (exact, then embedding match) | `_link_query_entities` |
| | Bridge Propagation | up to 3 hops along the query-most-similar sentences; finds bridge entities absent from the question | `_bridge_propagation` |
| | Bridged Candidate Generation | dense top-128 ∪ bridge passages (≤512) ∪ 2 passage-graph neighbours each | `_collect_candidates` |
| ② SP | Query-Induced Subgraph | candidates + their entities; entity-containment and passage-graph edges | `_subgraph_ppr` |
| | Subgraph PPR | restart = activated entities 75% + dense top-96 passages 25%, α = 0.85 | `_subgraph_ppr` |
| ③ AR | Score Fusion | PPR 0.60 / dense 0.18 / entity coverage 0.08 | `_fuse_scores` |
| | Graph-Aware Listwise Reranker | transformer over the top-128 fused candidates, graph features + pairwise attention bias, trained listwise | `spearrag/reranker.py` |

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
├── data -> ../handover/data   datasets (symlink)
└── requirements.txt
```

## Install

Python 3.9, CUDA 11.8 (tested on RTX A4000 16 GB). `pip install -r requirements.txt`, then
`python -m spacy download en_core_web_trf` (Medical also needs `en_core_sci_scibert` from scispaCy) and
`pip install cupy-cuda11x` so spaCy's transformer NER runs on the GPU (≈15x faster than CPU).
On this server everything is installed in the conda env `linearrag`:
`/root/anaconda3/envs/linearrag/bin/python`.

## Data

`data/` holds, per dataset, a training file for the reranker and the test release:

| dataset | reranker training | test (paper) | test with gold passage ids (R@5) |
|---|---|---|---|
| 2wiki | `2wiki_regimetrain.json` | `2wiki_linearrag_regime.json` | `2wiki_linearrag_regime_gold.json` |
| hotpotqa | `hotpotqa_regimetrain.json` | `hotpotqa_linearrag_regime.json` | `hotpotqa_linearrag_regime_gold.json` |
| musique | `musique_regimetrain.json` | `musique_linearrag_regime.json` | – |
| medical | `2wiki_regimetrain.json` (transfer) | `medical_linearrag_regime.json` | – |

Scalability corpora: `haystack_2wiki_{0,1m,5m,10m,20m}.json`, `scale_*.json`, `hotpotqa_fullwiki_regimeA.json`.
Format: `{"documents": [{"id", "text"}], "queries": [{"question", "answer", "gold_doc_ids"?, ...}]}`.

## Run

```bash
cd /workspace/GraphRAG/SpearRAG
PY=/root/anaconda3/envs/linearrag/bin/python
export OPENAI_API_KEY=...          # only needed for generation/judging

# main table (one row): ~10 min per seed per dataset, ~$0.95 per seed (Medical ~$2)
$PY scripts/run_spearrag.py --dataset 2wiki --seeds 42,43

# ablations: no_bridge | no_candidate_generation | no_subgraph_ppr | no_graph_features | no_rerank
$PY scripts/run_spearrag.py --dataset hotpotqa --ablation no_subgraph_ppr
$PY scripts/paired_test.py --ablations no_bridge no_candidate_generation no_subgraph_ppr no_graph_features no_rerank

# retrieval only ($0): R@5 / Complete@5 and timing; --ppr-scope global is the subgraph-vs-global control
$PY scripts/run_spearrag.py --dataset 2wiki --test-data-file 2wiki_linearrag_regime_gold.json --skip-generation
$PY scripts/run_spearrag.py --dataset 2wiki --test-data-file 2wiki_linearrag_regime_gold.json --skip-generation --ppr-scope global

# large corpora: fix the spaCy batch (a 16 GB GPU runs out of memory otherwise)
$PY scripts/with_spacy_batch.py 128 scripts/run_spearrag.py --dataset 2wiki --test-data-file haystack_2wiki_20m.json --skip-generation
$PY scripts/with_spacy_batch.py 128 scripts/measure_indexing.py --corpus data/scale_5m.json
```

Results go to `results/{dataset}_spearrag[_{ablation}][_global_ppr].json` (per-seed metrics, bootstrap CIs,
retrieved contexts, answers, per-question judge scores, OpenAI token usage, index/retrieval times).
For timing, run one job at a time on an otherwise idle machine and delete `--working-dir` first (caches).

Library use:

```python
from sentence_transformers import SentenceTransformer
from spearrag import ProjectConfig, SpearRAGRetriever
from spearrag.reranker import (RerankerConfig, FeatureExtractor, AdaptiveReranker, build_examples,
                               create_reranker, train_reranker, doc_index_gold_resolver, seed_everything)

emb = SentenceTransformer("all-mpnet-base-v2")
train_r = SpearRAGRetriever(ProjectConfig("train", emb)); train_r.index(train_passages)
test_r = SpearRAGRetriever(ProjectConfig("test", emb));   test_r.index(test_passages)
cfg = RerankerConfig()
ext = FeatureExtractor(train_r, cfg)
resolve = doc_index_gold_resolver({doc_id: i for i, doc_id in enumerate(train_doc_ids)})
tr, _ = build_examples(ext, train_rows, training=True, gold_resolver=resolve)
va, _ = build_examples(ext, valid_rows, training=False, gold_resolver=resolve)
seed_everything(cfg.seed); model = create_reranker(train_r, cfg); train_reranker(model, tr, va, cfg)
top5 = AdaptiveReranker(test_r, model, cfg).retrieve([{"question": "..."}])[0]["sorted_passage"]
```
Passages are indexed as `"{i}:{text}"`; the leading index is how gold ids and R@5 are resolved.

## Reproducibility notes

* **Equivalence with the research code.** The paper's numbers were produced by the research folder
  (`handover/`: `commurag/qacf_rag.py`, `commurag/qacf_listwise_reranker.py`, `final_test_eval.py --no-community`).
  This package is that configuration with the unused community machinery removed. `scripts/check_equivalence.py`
  indexes a corpus with both and compares every question's candidates, PPR, coverage, fused scores, fused ranking
  and reranker inputs; see `results/equivalence_2wiki.json`.
* **Reranker input width.** Under `--no-community` the research reranker received three community features as
  constant zeros (`community_route_z`, `inverse_community_size`, pair `same_community`). They are removed here
  (11 scalar + 4 pair features instead of 13 + 5). A zero input contributes nothing to the network, so the model
  family is unchanged; checkpoints from the research code are not loadable here.
* **Seeding.** The research script created the reranker before seeding, so its initial weights were not
  controlled by `--seeds`: on 2Wiki two runs with identical reranker inputs gave R@5 91.63 vs 91.64 while 16% of
  top-5 sets differed. `run_spearrag.py` seeds before model creation; whether seeded GPU runs are then bit-identical
  has not been verified, so report several seeds.
* **Generation prompt and judge** are copied unchanged from the research script (CoT prompt, 0/1 gpt-4o-mini judge).
* **Paper-run deviation:** the original fused-candidate truncation (only when > 4096 candidates) included a
  community bonus; it can never trigger here (dense 128 + bridge 512 + neighbours ≤ 1,280 < 4,096) and is omitted.

## Known property worth checking

With the paper's listwise-only objective, the reranker's pairwise head (`pair_query`, `pair_key`, `pair_edge`)
receives no gradient, yet greedy selection adds `0.35 × sigmoid(pair logit)` after the first pick. That term
therefore comes from the head's initial weights. Training it (`pair_loss_weight > 0`) or setting
`pair_selection_weight = 0` are both untested here.
