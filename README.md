# VibeLlama Model Evaluation

This repository contains scripts and configuration for evaluating the fine-tuned and base Llama-3.2 models on sentiment analysis. It runs generative inference (via `model.generate()`) using the exact chat‐prompt format you trained on, measures accuracy, latency, throughput, and resource usage, and outputs a tidy CSV for downstream statistical analysis.

---

## 📂 Repository Structure

```
.
|__ evaluate_models.py        # Main evaluation script
|__ inference-config.json     # Configuration for model specs & evaluation
|__ requirements.txt          # Python dependencies
|__ README.md                 # This file
```

---

## 🔧 Prerequisites

- **Python 3.8+**  
- **CUDA-enabled GPU** (e.g. A100 40 GB)  
- Conda or virtualenv to isolate the environment  
- **Hugging Face Transformers** with bitsandbytes support  
- **IMDb test split** (public through `datasets`) or a custom CSV

Install dependencies:

```bash
pip install torch transformers bitsandbytes datasets pandas numpy psutil tqdm scikit-learn
```

---

## ⚙️ Configuration (`inference-config.json`)

Create or edit `inference-config.json` with the experiment settings. Example:

```json
{
  "model_sizes":        [1, 3, 11],
  "seeds":              [42, 123, 456, 789, 555],
  "model_hub_namespace":"hf-username",
  "test_file":          "",                // path to CSV with columns `text,label` or leave empty to use IMDb test
  "max_new_tokens":     50,
  "n_reps":             3
}
```

- **model_sizes**: Sizes (in billions) to evaluate (1, 3, 11)  
- **seeds**: The 5 fine-tune seeds per size  
- **model_hub_namespace**: HF repo namespace for VibeLlama checkpoints  
- **test_file**: (optional) path to a CSV with columns `text,label`  
- **max_new_tokens**: Generation cap per prompt (set ≥20)  
- **n_reps**: Repetitions per model run for mean±std

---

## 🚀 Usage

1. Make sure the `inference-config.json` is populated.  
2. Set the HF token environment variable, `export HF_TOKEN=hf_…`.  
3. Run the evaluation:

   ```bash
   python3 evaluate_models.py
   ```

4. When complete, a `results.csv` will be generated with one summary row per model × repetition (21 specs × n_reps rows).

---

## 📊 Output (`results.csv`)

Each row contains:

| Column           | Description                                                 |
|------------------|-------------------------------------------------------------|
| model_name       | HF model ID                                                 |
| size             | Model size in billions                                      |
| seed             | Fine-tune seed (blank for base models)                      |
| quant            | `"fp32"` or `"4bit"`                                        |
| rep              | Repetition index (1…n_reps)                                 |
| accuracy         | Fraction of correct predictions on test set                 |
| f1               | F1‐score (binary)                                           |
| throughput       | Samples processed per second                                |
| latency          | Average per‐sample generation time (s)                      |
| gpu_peak_mem_mb  | Peak GPU memory (MB) during inference                       |
| cpu_rss_mb       | Host‐RAM RSS (MB)                                           |
| param_count      | Number of parameters                                        |
| param_mem_mb     | Model parameter memory footprint (MB)                       |

---

## 💡 Tips

- **Batch vs. single-sample**: This script uses batch_size=1 to measure per-sample latency. To measure batched throughput, adapt the code to batch inputs.  
- **Error handling**: If one model fails to load or generate, wrap `evaluate_one()` calls in `try/except` to continue.  
- **Statistical analysis**: Use `results.csv` in R, Python (statsmodels), or Excel to perform ANOVA, t-tests, and produce plots (accuracy vs. size, throughput vs. size, etc.).

---

## 📜 License

MIT License.
