from datasets import load_dataset, Dataset
import polars as pl
import os

def dataset_from_polars(polars_df):
    NUM_PROC = os.cpu_count()
    train = (
        Dataset.from_polars(polars_df)
        .filter(  # To create an anchor, positive, and negative structure, delete rows where the positive and negative are identical.
            lambda example: example["MisconceptionID"] != example["PredictMisconceptionId"],
            num_proc=4,
        )
    )
    return train
