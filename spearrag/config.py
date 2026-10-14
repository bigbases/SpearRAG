"""Runtime settings shared by every SpearRAG component (models, cache location, batching)."""
from dataclasses import dataclass
from typing import Any


@dataclass
class ProjectConfig:
    dataset_name: str               # names the cache sub-directory under working_dir
    embedding_model: Any            # a sentence-transformers model (all-mpnet-base-v2 in the paper)
    working_dir: str = "./working"  # NER / embedding caches live here
    spacy_model: str = "en_core_web_trf"  # en_core_sci_scibert for Medical
    ner_n_process: int = 8          # spaCy processes when NER runs on CPU (ignored on GPU)
    batch_size: int = 128           # embedding batch size
    max_workers: int = 16           # spaCy batch = len(passages) // max_workers
