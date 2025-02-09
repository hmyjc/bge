import pandas as pd
from copy import deepcopy
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

def data_process(df):
    df = deepcopy(df)
    grouped = df.groupby("QuestionId")

    question_dict = {}
    for question_id, group in grouped:
        question_data = group.to_dict(orient="records")[0]
        del question_data["QuestionId"]
        question_dict[question_id] = question_data

    all_questions = list(question_dict.keys())

    queries = []

    for qid in all_questions:
        info = question_dict[qid]

        for answer_key in ["A", "B", "C", "D"]:
            if info["CorrectAnswer"] == answer_key:
                continue
            this_example = dict()
            this_key = f"{qid}_{answer_key}"
            this_example["QuestionId_Answer"] = this_key
            this_example["Option"] = answer_key

            # ---
            for col in ["SubjectName", "ConstructName", "QuestionText"]:
                this_example[col] = info[col]

            this_example["CorrectAnswerText"] = info[f"Answer{info['CorrectAnswer']}Text"]
            this_example["AnswerText"] = info[f"Answer{answer_key}Text"]
            this_example["MisconceptionID"] = info[f"Misconception{answer_key}Id"]
            this_example["AllOptionText"] = "\n- ".join([info[f"Answer{x}Text"] for x in ["A", "B", "C", "D"]])
            this_example["AllOptionText"] = f"\n- {this_example['AllOptionText']}"
            queries.append(this_example)

    query_df = pd.DataFrame(queries).dropna()
    return query_df

def create_training_text(row):
    text = f"""
    {row["ConstructName"]}
    {row["QuestionText"]}
    Answer: {row["AnswerText"]}
    Misconception: {row["Misconception"]}
    """
    return text
def get_model(cfg):
    model = SentenceTransformer(cfg.MODEL_NAME)
    return model
def retrieve(model,train,label_df):
    train_long_vec = model.encode(
    train["FullText"].values, normalize_embeddings=True
)
    misconception_mapping_vec = model.encode(
    label_df["MisconceptionName"].values, normalize_embeddings=True
)
    train_cos_sim_arr = cosine_similarity(train_long_vec, misconception_mapping_vec)
    train_sorted_indices = np.argsort(-train_cos_sim_arr, axis=1)
    print('train_sorted_indices.shape',train_sorted_indices.shape)
    return train_sorted_indices
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config_path)
    print(cfg)
    df_train_dir = os.path.join(cfg.DATA_PATH,"train.csv")
    df_train = pd.read_csv(df_train_dir)   
    train_dir=os.path.join(cfg.DATA_PATH,"output.parquet")
    train = pd.read_parquet(train_dir)
    misconception_mapping_dir = os.path.join(cfg.DATA_PATH,"misconception_mapping.csv")
    misconception_mapping = pd.read_csv(misconception_mapping_dir)
    
    mapping = {}
    for k, v in zip(misconception_mapping["MisconceptionId"].values, misconception_mapping["MisconceptionName"].values):
        mapping[k] = v

    df=data_process(df_train)
    df.to_parquet(os.path.join(cfg.DATA_PATH,"df_process.parquet"))

    train["MisconceptionID"] = df["MisconceptionID"].values.astype(int)
    train["GroundTruthMisconception"] = train["MisconceptionID"].apply(lambda x: mapping[x])
    train["FullText"] = train.apply(lambda row: create_training_text(row), axis=1)

    model=get_model(cfg)
    train_sorted_indices=retrieve(model,train,misconception_mapping)
    train["PredictMisconceptionId"] = train_sorted_indices[:, :cfg.RETRIEVE_NUM].tolist()
    train_exploded = train.explode("PredictMisconceptionId")
    train_exploded["PredictMisconception"] = train_exploded["PredictMisconceptionId"].apply(lambda x:mapping[x])

    output_dir=os.path.join(cfg.DATA_OUTPUT_PATH,"train_exploded.parquet")
    train_exploded.to_parquet(output_dir)
