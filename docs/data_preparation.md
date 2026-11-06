# Data Preparation

Download the training annotations (`OpenView_dataset.json`) and test benchmark annotations (`OpenView_bench.json`) from our [Hugging Face repository](https://huggingface.co/datasets/openview2026/OpenView_data), which also provides image preparation instructions and preprocessing scripts. **Source images** must be obtained from the original providers under their respective terms of use.

Arrange the files as follows:

```text
OpenView/
├── annotations/
│   ├── OpenView_dataset.json
│   └── OpenView_bench.json
└── data/
    ├── 360loc/
    ├── 360x/
    ├── 3601m/
    ├── mapillary/
    ├── mp3d/
    └── test/
```

For benchmark inference and evaluation, our code requires only source panoramas in `data/test/` and `annotations/OpenView_bench.json`; no additional conversion is needed.

## [Optional] Conversion for training

The conversion script projects question views from panoramas and converts VQA annotations into image–conversation pairs for supervised fine-tuning. For training, see the [Qwen3-VL training framework](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune).

From the repository root:

```bash
cd data_preparation
python convert_to_format.py \
  --input_file ../annotations/OpenView_dataset.json \
  --output_file ../annotations/OpenView_dataset_converted.json
```
