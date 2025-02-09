from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
def get_model(cfg):
    model = SentenceTransformer(cfg.MODEL_NAME)
    return model
