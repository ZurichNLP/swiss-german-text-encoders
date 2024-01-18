# Adapted from https://github.com/ZurichNLP/swissbert/tree/master/evaluation/romansh_alignment

from dataclasses import dataclass
from typing import Union, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, CanineConfig, AutoModel, AutoTokenizer

from evaluation.retrieval.retrieval_datasets import NTREXDataset
from modeling.config import CharXmodConfig
from modeling.model import CharXmodModel
from modeling.tokenization import CharXmodTokenizer


class LayerAggregation(str):
    SINGLE = "single"
    AVERAGE = "average"


class SentenceEncoder:

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer,
                 aggregation: Union[str, LayerAggregation] = LayerAggregation.AVERAGE,
                 ):
        self.model = model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.tokenizer = tokenizer
        self.aggregation = aggregation
        self._is_canine = isinstance(self.model.config, (CanineConfig, CharXmodConfig))

    def embed_tokens(self, text: str, lang_id: int = None) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=False,
        ).to(self.model.device)
        with torch.no_grad():
            model_args = {
                "output_hidden_states": True,
                "return_dict": True,
            }
            if lang_id is not None:
                model_args["lang_ids"] = torch.tensor([lang_id]).to(self.model.device)
            output = self.model(**inputs, **model_args)

            if self.aggregation == LayerAggregation.SINGLE:
                if self._is_canine:
                    embeddings = output.hidden_states[-3].squeeze(0)  # Skip character layers
                else:
                    embeddings = output.last_hidden_state.squeeze(0)
            elif self.aggregation == LayerAggregation.AVERAGE:
                if self._is_canine:
                    embeddings = torch.cat(output.hidden_states[2:-2], dim=0).mean(dim=0)  # Skip character layers
                else:
                    embeddings = torch.cat(output.hidden_states, dim=0).mean(dim=0)
            else:
                raise ValueError(f"Invalid aggregation: {self.aggregation}")
        return embeddings

    def __str__(self):
        return f"HuggingfaceEncoder({self.model.name_or_path.replace('/', '_')}, aggregation={self.aggregation})"


@dataclass
class SentenceAlignmentResult:
    num_sentences: int
    accuracy: float


class SentenceAlignmentBenchmark:

    def __init__(self, src_sentences: List[str], tgt_sentences: List[str]):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        assert len(self.src_sentences) == len(self.tgt_sentences)

    def evaluate_encoder_bert_score(self,
                                    encoder: SentenceEncoder,
                                    src_lang_id: int = None,
                                    tgt_lang_id: int = None,
                                    top_k: int = 1,
                                    batch_size: int = 8,
                                    device=None,
                                    ) -> SentenceAlignmentResult:
        print(f"Encoding tokens with {encoder}...")
        src_embeddings = []
        for sentence in tqdm(self.src_sentences):
            src_embeddings.append(encoder.embed_tokens(sentence, lang_id=src_lang_id))
        tgt_embeddings = []
        for sentence in tqdm(self.tgt_sentences):
            tgt_embeddings.append(encoder.embed_tokens(sentence, lang_id=tgt_lang_id))
        print("Calculating accuracy...")
        accuracy = self._calculate_bert_score_accuracy(src_embeddings, tgt_embeddings,
                                                             top_k=top_k, device=(device or encoder.model.device),
                                                             batch_size=batch_size)
        result = SentenceAlignmentResult(
            num_sentences=len(self.src_sentences),
            accuracy=accuracy,
        )
        return result

    def _calculate_bert_score_accuracy(self, all_query_embeddings, document_embeddings, top_k=1, batch_size=8, device="cpu") -> float:
        correct = 0
        for i, query_embeddings in enumerate(tqdm(all_query_embeddings)):
            scores = []
            # Batch document_embeddings to reduce memory usage
            for j in range(0, len(document_embeddings), batch_size):
                scores.extend(bert_score(query_embeddings.cpu().numpy(), document_embeddings[j:j+batch_size], device=device))
            scores = np.array(scores)
            correct += i in scores.argsort()[-top_k:]
        return correct / len(all_query_embeddings)


def bert_score(
        query_embeddings: np.ndarray,
        document_embeddings: Union[List[np.ndarray], np.ndarray],
        device: str = "cpu",
):
    """
    Adapted from https://github.com/Tiiiger/bert_score/blob/cb582ed5c88b02230b8f101173fd959b68023dc6/bert_score/utils.py#L469
    """
    assert document_embeddings[0].shape[-1] == query_embeddings.shape[-1]
    if isinstance(document_embeddings, list):
        # Pad document_embeddings to the same length with zeros
        max_length = max(len(embeddings) for embeddings in document_embeddings)
        document_embeddings = [np.pad(embeddings.cpu().numpy(), ((0, max_length - len(embeddings)), (0, 0)), 'constant') for embeddings in document_embeddings]
        document_embeddings = np.array(document_embeddings)

    with torch.no_grad():
        ref_embedding = torch.from_numpy(query_embeddings).unsqueeze(0).repeat(len(document_embeddings), 1, 1).to(device)
        hyp_embedding = torch.from_numpy(document_embeddings).to(device)

        ref_masks = (ref_embedding != 0).all(-1)
        hyp_masks = (hyp_embedding != 0).all(-1)
        # Avoid NaN
        ref_embedding[~ref_masks] = 1
        hyp_embedding[~hyp_masks] = 1

        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

        batch_size = ref_embedding.size(0)
        sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
        masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
        masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

        masks = masks.float().to(sim.device)
        sim = sim * masks

        word_precision = sim.max(dim=2)[0]
        word_recall = sim.max(dim=1)[0]

        P = word_precision.sum(dim=1) / hyp_masks.sum(dim=1)
        R = word_recall.sum(dim=1) / ref_masks.sum(dim=1)
        F = 2 * P * R / (P + R)
        return F.cpu().numpy()


def main(
        model_name_or_path: str,
        gsw_fallback_de: bool = False,
):
    AutoModel.register(CharXmodConfig, CharXmodModel)
    model = AutoModel.from_pretrained(model_name_or_path)
    if model.config.model_type == "char_xmod":
        tokenizer = CharXmodTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    encoder = SentenceEncoder(
        model=model,
        tokenizer=tokenizer,
    )
    dataset = NTREXDataset()
    lang_pairs = [
        ("deu", "gsw-BE"),
        ("deu", "gsw-ZH"),
    ]
    lang_map = {
        "deu": "de_CH",
        "gsw-BE": "gsw" if not gsw_fallback_de else "de_CH",
        "gsw-ZH": "gsw" if not gsw_fallback_de else "de_CH",
    }

    if model_name_or_path == "facebook/xmod-base":
        for key, value in list(lang_map.items()):
            if value == "de_CH":
                lang_map[key] = "de_DE"

    for src_lang, tgt_lang in lang_pairs:
        print(f"Language pair: {src_lang} -> {tgt_lang}")
        src_sentences = dataset.get_sentences(src_lang)
        tgt_sentences = dataset.get_sentences(tgt_lang)
        benchmark = SentenceAlignmentBenchmark(src_sentences, tgt_sentences)
        if hasattr(model.config, "languages"):
            src_lang_id = model.config.languages.index(lang_map[src_lang])
            tgt_lang_id = model.config.languages.index(lang_map[tgt_lang])
        else:
            src_lang_id = None
            tgt_lang_id = None
        print(f"src_lang: {lang_map[src_lang]}; tgt_lang: {lang_map[tgt_lang]}; src_lang_id: {src_lang_id}; tgt_lang_id: {tgt_lang_id}")
        result = benchmark.evaluate_encoder_bert_score(
            encoder=encoder,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
        )
        print(f"Accuracy: {result.accuracy}")
        print()
