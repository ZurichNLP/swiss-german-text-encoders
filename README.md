Code for the paper ["Modular Adaptation of Multilingual Encoders to Written Swiss German Dialect"](https://aclanthology.org/2024.moomin-1.3/)

**[Blog post](https://vamvas.ch/swiss-german-encoder)**

List of models released for this paper:
* [hf.co/ZurichNLP/swissbert](https://hf.co/ZurichNLP/swissbert)
  * this paper adds a Swiss German adapter
  * the original version of SwissBERT is accessible at [model revision v1](https://huggingface.co/ZurichNLP/swissbert/tree/v1))
* [hf.co/ZurichNLP/swiss-german-xlm-roberta-base](https://hf.co/ZurichNLP/swiss-german-xlm-roberta-base)
* [hf.co/ZurichNLP/swiss-german-canine](https://hf.co/ZurichNLP/swiss-german-canine)
* [hf.co/ZurichNLP/swiss-german-swissbert-char](https://hf.co/ZurichNLP/swiss-german-swissbert-char)


## Installation

- Requirements: Python >= 3.8, PyTorch
- `pip install -r requirements.txt`

## Continued Pre-training

### Data

- Not all the data we used are publicly available. See `data/README.md` for details.
- `python -m scripts.preprocess_continued_pretraining_data`

### Training

- Subword level: `python -m scripts.continued_pretraining_subword <model_name_or_path>`
  - Tested with `xlm-roberta-base`, `facebook/xmod-base`, `ZurichNLP/swissbert`
- Character level: `python -m scripts.continued_pretraining_char <model_name_or_path>`
  - Tested with `google/canine-s`, `facebook/xmod-base`, `ZurichNLP/swissbert` (the latter two correspond to the GLOBI approach described in Section 4.3 of the paper)

## Evaluation

### Data
- See `data/README.md` for instructions on how to download the data.

### Fine-tuning and testing

- Part-of-speech tagging: `python -m scripts.evaluate_pos <model_name_or_path>`
- German dialect identification: `python -m scripts.evaluate_gdi <model_name_or_path>`
- Retrieval (no fine-tuning): `python -m scripts.evaluate_retrieval <model_name_or_path>`

## License
- This code repository: MIT license
- Model weights: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

## Citation

```bibtex
@inproceedings{vamvas-etal-2024-modular,
    title = "Modular Adaptation of Multilingual Encoders to Written {S}wiss {G}erman Dialect",
    author = {Vamvas, Jannis  and
      Aepli, No{\"e}mi  and
      Sennrich, Rico},
    editor = {V{\'a}zquez, Ra{\'u}l  and
      Mickus, Timothee  and
      Tiedemann, J{\"o}rg  and
      Vuli{\'c}, Ivan  and
      {\"U}st{\"u}n, Ahmet},
    booktitle = "Proceedings of the 1st Workshop on Modular and Open Multilingual NLP (MOOMIN 2024)",
    month = mar,
    year = "2024",
    address = "St Julians, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.moomin-1.3",
    pages = "16--23"
}
```
