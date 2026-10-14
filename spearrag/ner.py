import os
import spacy

# --- GPU enablement (added for this project) -------------------------------
# en_core_web_trf is a transformer NER model. Without an explicit
# spacy.prefer_gpu() call BEFORE spacy.load(), spaCy silently runs it on CPU:
# prefer_gpu() returns False when cupy is missing and no warning is emitted.
# That made trf NER ~15x slower than necessary (measured: 525K words still
# unfinished after 9 min at 808% CPU / 0% GPU).
# Requires cupy-cuda11x (installed in the linearrag env).
# NOTE: when the GPU is active, spaCy's nlp.pipe MUST run with n_process=1 --
# multiprocessing + CUDA in the same pipe call deadlocks or falls back.
_SPACY_ON_GPU = False
try:
    _SPACY_ON_GPU = bool(spacy.prefer_gpu())
except Exception:
    _SPACY_ON_GPU = False
# ---------------------------------------------------------------------------
from collections import defaultdict


class SpacyNER:
    def __init__(self, spacy_model: str):
        self.nlp = spacy.load(spacy_model)

    def batch_ner(self, hash_id_to_passage: dict, max_workers: int, n_process: int = 1):
        # cupy's "current device" is THREAD-LOCAL. the retriever may run indexing NER inside a
        # ThreadPoolExecutor (retriever.py index() -> ner_future.result()), so the
        # worker thread starts with no device bound and spacy-transformers' cupy
        # ops raise `cudaErrorInvalidDevice: invalid device ordinal`. LinearRAG and
        # EHRAG never hit this because they call batch_ner on the main thread.
        # Binding the device here is idempotent and cheap.
        if _SPACY_ON_GPU:
            try:
                import cupy
                # Device 0 was hardcoded here, which silently pinned spaCy to GPU 0
                # no matter what --gpu the caller asked for. Two runs on
                # "different" GPUs then both put their transformer NER on GPU 0 and
                # died with CUDA OOM (2026-09-21, twice). SPEARRAG_SPACY_GPU is set from
                # --gpu by scripts/run_spearrag.py; the default is GPU 0.
                cupy.cuda.Device(int(os.environ.get("SPEARRAG_SPACY_GPU", "0"))).use()
            except Exception:
                pass
        passage_list = list(hash_id_to_passage.values())
        batch_size = max(1, len(passage_list) // max(max_workers, n_process))
        docs = self.nlp.pipe(passage_list, batch_size=batch_size, n_process=(1 if _SPACY_ON_GPU else n_process))

        passage_hash_id_to_entities = {}
        sentence_to_entities = defaultdict(list)

        for idx, doc in enumerate(docs):
            hash_id = list(hash_id_to_passage.keys())[idx]
            unique_entities = set()
            for ent in doc.ents:
                if ent.label_ in ("ORDINAL", "CARDINAL"):
                    continue
                sent_text = ent.sent.text
                ent_text = ent.text
                if ent_text not in sentence_to_entities[sent_text]:
                    sentence_to_entities[sent_text].append(ent_text)
                unique_entities.add(ent_text)
            passage_hash_id_to_entities[hash_id] = list(unique_entities)

        return passage_hash_id_to_entities, sentence_to_entities

    def question_ner(self, question: str) -> set:
        doc = self.nlp(question)
        return {
            ent.text.lower()
            for ent in doc.ents
            if ent.label_ not in ("ORDINAL", "CARDINAL")
        }
