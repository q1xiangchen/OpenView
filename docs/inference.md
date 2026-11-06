# Benchmark Inference

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The benchmark inference and evaluation environment (Transformers 4.57.1) is separate from the [pipeline](pipeline.md) environment (vLLM). Optionally install `flash-attn==2.7.4.post1 --no-build-isolation` for faster inference; without it, the scripts fall back to PyTorch SDPA attention.

Prepare the benchmark panoramas and annotations following [Data Preparation](data_preparation.md). Question views are projected on the fly during inference.

## Local Inference with Transformers

From the repository root:

```bash
cd inference
python inference_qwen_vl.py \
  --model_path /path/to/qwen3_model \
  --test_gt_path ../annotations/OpenView_bench.json \
  --data_path ../data/test \
  --output_json ../results/qwen_output.json
```

Use `inference_internvl.py` or `inference_llavanext.py` for InternVL or LLaVA-NeXT, with the same arguments. For Qwen, the model path must contain `qwen2.5` / `qwen2_5` or `qwen3` (case-insensitive, e.g. `Qwen/Qwen2.5-VL-7B-Instruct`) to select the corresponding model class.

## API Inference

From the repository root:

```bash
cd inference
export OPENAI_API_KEY="your-key"

python inference_api.py \
  --model gpt-4o \
  --test_gt_path ../annotations/OpenView_bench.json \
  --data_path ../data/test \
  --output_json ../results/gpt-4o_output.json
```

For Gemini, set `GEMINI_API_KEY` and pass the Gemini model name. For a local vLLM server, pass its model name and `--base_url`.

For scoring predictions, see [Evaluation](evaluation.md). To construct new VQAs, see [Pipeline](pipeline.md).
