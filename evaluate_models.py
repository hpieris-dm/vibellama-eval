#!/usr/bin/env python3
"""
evaluate_models.py

Generative inference using SFT-tuned causal Llama models,
measuring accuracy, performance, and resource usage.
Outputs an incremental CSV for statistical analysis.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def load_config(path="inference-config.json"):
    with open(path) as f:
        return json.load(f)

def format_inference_prompt(review: str) -> str:
    system_prompt = "You are a helpful assistant that analyzes the sentiment of provided inputs."
    if len(review) > 1000:
        review = review[:1000] + "..."
    return (
        "<|begin_of_text|><|system|>\n"
        f"{system_prompt}<|end_of_text|>\n"
        "<|user|>\n"
        "Determine if the following input has a positive or negative sentiment. "
        "Reply with only 'positive' or 'negative'.\n\n"
        f"Review: {review}<|end_of_text|>\n"
        "<|assistant|>\n"
    )

def make_model_specs(cfg):
    specs = []
    for size in cfg["model_sizes"]:
        for seed in cfg["seeds"]:
            specs.append({
                "model_name": f"{cfg['model_hub_namespace']}/VibeLlama-{size}b-seed-{seed}",
                "size": size,
                "seed": seed,
                "quant": "4bit"
            })
    for size in cfg["model_sizes"]:
        base_id = (
            "meta-llama/Llama-3.2-11B-Vision-Instruct"
            if size == 11 else
            f"meta-llama/Llama-3.2-{size}B-Instruct"
        )
        specs.append({"model_name": base_id, "size": size, "seed": None, "quant": "bf16"})
        specs.append({"model_name": base_id, "size": size, "seed": None, "quant": "4bit"})
    return specs

def load_causal_model(model_name, quant, seed):
    """
    - If seed is not None, model_name is the adapter repo: load base + attach adapter.
    - If seed is None, model_name is the full base checkpoint.
    """
    # 1) Base loading path
    if seed is None:
        # Base model only
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
        elif quant == "bf16":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.bfloat16
            )
        else:
            raise ValueError(f"Unknown quant mode for base model: {quant}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    else:
        # Adapter path: parse base_id from model_name
        size = int(model_name.split("-")[1][:-1])  # e.g. "3b"→3
        if size == 11:
            base_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        else:
            base_id = f"meta-llama/Llama-3.2-{size}B-Instruct"

        # 2a) Load base model in the right precision/quant
        if quant == "4bit":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_id, device_map="auto", quantization_config=bnb
            )
        elif quant == "bf16":
            base_model = AutoModelForCausalLM.from_pretrained(
                base_id, device_map="auto", torch_dtype=torch.bfloat16
            )
        else:
            raise ValueError(f"Unknown quant mode for adapter model: {quant}")

        # 2b) Attach the LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            model_name,       # this is the adapter repo
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model.eval()
    return model, tokenizer


def get_sentiment_prediction(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    torch.cuda.reset_peak_memory_stats(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    t1 = time.perf_counter()

    gen_ids = generation_output[0, input_ids.shape[-1]:]

    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    mem = torch.cuda.max_memory_reserved(model.device) / 1024**2
    latency = t1 - t0

    lbl = 1 if response.lower().startswith("positive") else 0

    return lbl, latency, response, mem


def evaluate_one(spec, reviews, labels, cfg):
    n_reps = cfg.get("n_reps", 3)
    max_new_tokens = cfg.get("max_new_tokens", 50)
    rows = []

    for rep in range(1, n_reps+1):
        # load
        model, tokenizer = load_causal_model(spec["model_name"], spec["quant"], spec["seed"])

        # param footprint
        param_count = sum(p.numel() for p in model.parameters())
        param_mem_mb = sum(p.numel()*p.element_size() for p in model.parameters())/1024**2

        # warm-up
        _ = get_sentiment_prediction(model, tokenizer, format_inference_prompt(reviews[0]), max_new_tokens)

        preds, times, mems = [], [], []
        for review in tqdm(reviews, desc=f"{spec['model_name']} rep{rep}", leave=False):
            lbl, t, _, m = get_sentiment_prediction(
                model, tokenizer, format_inference_prompt(review), max_new_tokens
            )
            preds.append(lbl)
            times.append(t)
            mems.append(m)

        # metrics
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

        # cleanup
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return rows

def main():
    cfg = load_config()
    specs = make_model_specs(cfg)

    # load test set or CSV
    if cfg.get("test_file") and os.path.exists(cfg["test_file"]):
        df = pd.read_csv(cfg["test_file"])
        reviews, labels = df["text"].tolist(), df["label"].tolist()
    else:
        ds = load_dataset("imdb", split="test")
        reviews = ds["text"]
        labels  = ds["label"]

        # stratified subsample
        sub_n = cfg.get("subsample_n", 2500)
        reviews, _, labels, _ = train_test_split(
            reviews, labels,
            stratify=labels,
            train_size=sub_n,
            random_state=cfg.get("seed", 42)
        )

    out_csv = "results.csv"
    # remove existing file so we can write headers fresh
    if os.path.exists(out_csv):
        os.remove(out_csv)

    for spec in specs:
        print(f">>> Evaluating {spec['model_name']} ({spec['quant']}) …")
        try:
            rows = evaluate_one(spec, reviews, labels, cfg)
        except Exception as e:
            print(f"[!] ERROR evaluating {spec['model_name']}: {e}")
            continue

        # append this model’s rows to CSV
        df_rows = pd.DataFrame(rows)
        df_rows.to_csv(
            out_csv,
            mode="a",
            index=False,
            header=not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
        )
        print(f"[✓] Appended {len(rows)} rows for {spec['model_name']}")

    print("✅ All done. Results in", out_csv)

if __name__ == "__main__":
    main()
