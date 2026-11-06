# OpenView Data Construction Pipeline

Construct new VQAs from source panoramas using [vLLM](https://github.com/vllm-project/vllm) and `Qwen/Qwen2.5-VL-72B-Instruct`.

Install dependencies from the repository root in a separate environment from the [benchmark inference](inference.md) one, and prepare the images following [Data Preparation](data_preparation.md):

```bash
pip install -r requirements_pipeline.txt
```

## 1. Filter Panoramas

Configure paths and GPU settings at the top of `inference/batch_stage1.py`. The source CSV must contain a `file_name` column. Images are loaded from `data/<filename-prefix>/<filename>`, where the prefix is the text before the first underscore.

From the repository root:

```bash
cd inference
python batch_stage1.py
```

Review the judgments saved to `output_jsonl` and prepare a filtered CSV with panorama filename and scene label as its first two columns, including a header row. Exclude benchmark scenes and source videos.

## 2. Generate and Refine VQAs

Edit `inference/configs/batch_config.yaml`:

- Set `data_dir`, `out_dir`, and `cache_dir` for images and intermediate outputs.
- Set `annotation_path` to the filtered panorama CSV.
- Set `save_path` for the final refined VQA JSON (default: `../output/OpenView_generated_vqa.json`).
- Adjust `agent_config` and `batch_size` for your GPUs.

From `inference/`, run the visual analyzer, proposal generator, and proposal refiner in order:

```bash
python batch_stage2.py --config_path configs/batch_config.yaml
python batch_stage3.py --config_path configs/batch_config.yaml
python batch_stage4.py --config_path configs/batch_config.yaml
```

Stage 4 saves the refined VQA JSON to `save_path` in `configs/batch_config.yaml`. To prepare it for training, follow [Conversion for Training](data_preparation.md#optional-conversion-for-training).
