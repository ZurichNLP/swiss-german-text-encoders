import sys

from evaluation.retrieval.run_retrieval import main

model_name = sys.argv[1]

FALLBACK_DE = False  # Set to True to use Standard German adapter for Swiss German

main(
    model_name_or_path=model_name,
    gsw_fallback_de=FALLBACK_DE,
)
