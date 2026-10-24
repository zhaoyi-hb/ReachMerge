"""Evaluate an arbitrary state_dict with MergeRec's test routine (same metrics as merge_test.py).

Run from the MergeRec root:
  STATE_PATH=merged.pt python eval_state.py <TestMergeConfig arguments>
"""
import os
import sys

import torch
import tyro

sys.path.insert(0, os.getcwd())
from rec_retrieval.configs import NegativeSampleConfig, TestMergeConfig  # noqa: E402
from rec_retrieval.evaluator import Evaluator  # noqa: E402
from rec_retrieval.module.models import ModelType  # noqa: E402
from rec_retrieval.module.recommender import RecModule  # noqa: E402
from utils import test_model  # noqa: E402


def main():
    cfg = tyro.cli(TestMergeConfig)
    sd = torch.load(os.environ["STATE_PATH"], map_location="cpu")
    sd.pop("item_embeddings", None)
    model = ModelType[cfg.model_type].value(
        model_name_or_path=cfg.model_path, tokenizer_name_or_path=cfg.tokenizer_path,
        lora_config=(None if cfg.lora is not None and not cfg.lora.enable else cfg.lora),
        pooling_method=cfg.pooling_method, model_kwargs=cfg.model_kwargs.copy(),
        tokenizer_kwargs=cfg.tokenizer_kwargs.copy())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    module = RecModule(model=model, evaluator=Evaluator(metrics=cfg.metric_names, ks=cfg.ks),
                       negative_sample=NegativeSampleConfig(), similarity=cfg.similarity)
    test_model(module=module, model_type=ModelType[cfg.model_type], data_paths=cfg.data_paths,
               model_tokenizer=model.tokenizer, batch_size=cfg.batch_size, max_seq_len=cfg.max_seq_len,
               max_attribute_len=cfg.max_attribute_len, max_items=cfg.max_items, num_workers=cfg.num_workers,
               sequence_prompt=cfg.sequence_prompt, item_prompt=cfg.item_prompt,
               reverse_sequence=cfg.reverse_sequence, precision=cfg.precision, data_split=cfg.test_data_split,
               metrics_path=cfg.metrics_path, predictions_path=cfg.predictions_path,
               item_embeddings_path=cfg.item_embeddings_path, user_embeddings_path=cfg.user_embeddings_path)


if __name__ == "__main__":
    main()
