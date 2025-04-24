#!/usr/bin/env python3
"""
evaluate_models.py

Generative inference using SFT-tuned causal Llama models,
measuring accuracy, performance, and resource usage,
with the exact same chat-prompt format you used in training.
Outputs a CSV for statistical analysis.
"""

import os
import time
import gc
import json
import torch
import psutil
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from sklearn.metrics import f1_score

def load_config(path="inference-config.json"):
    with open(path) as f:
        return json.load(f)

def format_inference_prompt(review: str) -> str:
    """
    Mirrors the training prompt.
    """
    system_prompt = "You are a helpful assistant that analyzes the sentiment of provided inputs."
    if len(review) > 1000:
        review = review[:1000] + "..."
    prompt = (
        "<|begin_of_text|><|system|>\n"
        f"{system_prompt}<|end_of_text|>\n"
        "<|user|>\n"
        "Determine if the following input has a positive or negative sentiment. "
        "Reply with only 'positive' or 'negative'.\n\n"
        f"Review: {review}<|end_of_text|>\n"
        "<|assistant|>\n"
    )
    return prompt

def make_model_specs(cfg):
    specs = []
    # 1) The fine-tuned VibeLlama models (4-bit)
    for size in cfg["model_sizes"]:
        for seed in cfg["seeds"]:
            specs.append({
                "model_name": f"{cfg['model_hub_namespace']}/VibeLlama-{size}b-seed-{seed}",
                "size": size,
                "seed": seed,
                "quant": "4bit"
            })

    # 2) Base Llama models (FP32 & 4-bit), with special 11B name
    for size in cfg["model_sizes"]:
        if size == 11:
            base_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        else:
            base_id = f"meta-llama/Llama-3.2-{size}B-Instruct"
        specs.append({"model_name": base_id, "size": size, "seed": None, "quant": "fp32"})
        specs.append({"model_name": base_id, "size": size, "seed": None, "quant": "4bit"})
    return specs

def load_causal_model(model_name, quant):
    if quant == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", quantization_config=bnb
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model.eval()
    return model, tokenizer

def get_sentiment_prediction(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.reset_peak_memory_stats(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    t1 = time.perf_counter()
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    mem = torch.cuda.max_memory_reserved(model.device) / 1024**2
    text = response.lower()
    lbl = 1 if "positive" in text else 0
    return lbl, t1 - t0, response, mem

def evaluate_one(spec, reviews, labels, cfg):
    n_reps = cfg.get("n_reps", 3)
    max_new_tokens = cfg.get("max_new_tokens", 50)
    rows = []

    for rep in range(1, n_reps+1):
        # Load model + tokenizer
        model, tokenizer = load_causal_model(spec["model_name"], spec["quant"])

        # Parameter footprint
        param_count = sum(p.numel() for p in model.parameters())
        param_mem_mb = sum(p.numel()*p.element_size() for p in model.parameters()) / 1024**2

        # Warm-up on first prompt
        prompt0 = format_inference_prompt(reviews[0])
        _ = get_sentiment_prediction(model, tokenizer, prompt0, max_new_tokens)

        preds, times, mems = [], [], []
        for review in tqdm(reviews, desc=f"{spec['model_name']} rep{rep}", leave=False):
            prompt = format_inference_prompt(review)
            lbl, t, _, m = get_sentiment_prediction(model, tokenizer, prompt, max_new_tokens)
            preds.append(lbl)
            times.append(t)
            mems.append(m)

        accuracy = np.mean(np.array(preds) == np.array(labels))
        f1 = f1_score(labels, preds)
        throughput = len(reviews) / sum(times)
        latency = np.mean(times)
        gpu_peak = max(mems)
        cpu_rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2

        rows.append({
            "model_name": spec["model_name"],
            "size": spec["size"],
            "seed": spec["seed"] or "",
            "quant": spec["quant"],
            "rep": rep,
            "accuracy": accuracy,
            "f1": f1,
            "throughput": throughput,
            "latency": latency,
            "gpu_peak_mem_mb": gpu_peak,
            "cpu_rss_mb": cpu_rss,
            "param_count": param_count,
            "param_mem_mb": param_mem_mb
        })

        # Cleanup
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return rows

def main():
    cfg = load_config()
    specs = make_model_specs(cfg)

    # Loads a test CSV if provided, else defaults to IMDb test
    if cfg.get("test_file") and os.path.exists(cfg["test_file"]):
        df = pd.read_csv(cfg["test_file"])
        reviews = df["text"].tolist()
        labels  = df["label"].tolist()
    else:
        ds = load_dataset("imdb", split="test")
        reviews = ds["text"]
        labels  = ds["label"]

    all_rows = []
    for spec in specs:
        print(f">>> Evaluating {spec['model_name']} ({spec['quant']}) …")
        all_rows.extend(evaluate_one(spec, reviews, labels, cfg))

    pd.DataFrame(all_rows).to_csv("results.csv", index=False)
    print("✅ Saved results.csv")

if __name__ == "__main__":
    main()
