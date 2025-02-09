import argparse
from omegaconf import OmegaConf
import re
import vllm
import pandas as pd
def generate_prompt(row):
    sp="Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response>$$INSERT TEXT HERE$$</response>.\nBefore answering the question think step by step concisely in 1-2 sentence inside <thinking>$$INSERT TEXT HERE$$</thinking> tag and respond your final misconception inside <response>$$INSERT TEXT HERE$$</response> tag."
    Prompt=f"Question:{row['QuestionText']}\nIncorrect Answer:{row['AnswerText']}\nCorrect Answer:{row['Correct Answer']}\nConstruct Name:{row['ConstructName']}\nSubject Name:{row['ConstructName']}\n{sp}"
    return prompt
def extract_response(text):
    return ",".join(re.findall(r"<response>(.*?)</response>", text)).strip()
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config_path)
    print(cfg)
    df= pd.read_parquet("../competitions_data/df_process.parquet")
    # 应用函数并添加新列
    df['Prompt'] = df.apply(generate_prompt, axis=1)
    # 打印结果
    llm = vllm.LLM(
        cfg.LLM_MODEL,
        quantization="awq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95, 
        trust_remote_code=True,
        dtype="half", 
        enforce_eager=True,
        max_model_len=8192,
        disable_log_stats=True
    )
    tokenizer = llm.get_tokenizer()
    
    
    responses = llm.generate(
        df["Prompt"].values,
        vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0,  # randomness of the sampling
            seed=777, # Seed for reprodicibility
            skip_special_tokens=False,  # Whether to skip special tokens in the output.
            max_tokens=2048,  # Maximum number of tokens to generate per output sequence.
        ),
        use_tqdm = True
    )
    
    responses = [x.outputs[0].text for x in responses]
    df["FullResponse"] = responses
    
    responses = [extract_response(x) for x in responses]
    df["Misconception"] = responses
    df.to_parquet("../competitions_data/output.parquet", index=False)
