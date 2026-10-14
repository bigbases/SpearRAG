"""Run a script with spaCy's ``nlp.pipe`` batch size fixed (large corpora).

The NER batch defaults to len(passages) // max_workers, which grows with the corpus and runs a 16 GB GPU out
of memory around 10M tokens. All large-scale runs in the paper (haystack, fullwiki) used a fixed batch of 128:

    python scripts/with_spacy_batch.py 128 scripts/run_spearrag.py --dataset hotpotqa \
        --test-data-file hotpotqa_fullwiki_regimeA.json --skip-generation
"""
import runpy
import sys

batch = int(sys.argv[1])
script = sys.argv[2]
sys.argv = [script] + sys.argv[3:]

import spacy.language  # noqa: E402

_pipe = spacy.language.Language.pipe
spacy.language.Language.pipe = lambda self, texts, *a, **kw: _pipe(self, texts, *a, **{**kw, "batch_size": batch})
runpy.run_path(script, run_name="__main__")
