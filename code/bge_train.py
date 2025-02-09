import polars as pl
import re
import pandas as pd
import os
import numpy as np
import argparse
from omegaconf import OmegaConf
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from bge_dataset import dataset_from_polars
from bge_model import get_model
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config_path)
    
    train_exploded_dir = os.path.join(cfg.DATA_OUTPUT_PATH,"train_exploded.parquet")
    train_exploded = pd.read_parquet(train_exploded_dir)
    final_train = pl.from_pandas(train_exploded)
    
    train=dataset_from_polars(final_train)
    
    train = train.select_columns(["FullText", "GroundTruthMisconception", "PredictMisconception"])

    model = get_model(cfg)
    loss = MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=cfg.OUTPUT_PATH,
        # Optional training parameters:
        num_train_epochs=cfg.EPOCH,
        per_device_train_batch_size=cfg.BS,
        gradient_accumulation_steps=cfg.GRAD_ACC_STEP,
        per_device_eval_batch_size=cfg.BS,
        eval_accumulation_steps=cfg.GRAD_ACC_STEP,
        learning_rate=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        lr_scheduler_type="cosine_with_restarts",
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        logging_steps=100,
        report_to=REPORT_TO,  # Will be used in W&B if `wandb` is installed
        run_name=cfg.EXP_NAME,
        do_eval=False
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train,
        loss=loss
    )
    
    trainer.train()
    model.save_pretrained(cfg.MODEL_OUTPUT_PATH)
