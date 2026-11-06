from vllm import LLM, SamplingParams

from pathlib import Path
import jsonlines
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd
from utils.parser import parser

######## modify the following variables to fit your own situation ########
data_root = Path("../data")
meta_file = Path("../annotations/OpenView_dataset_sources.csv")
output_jsonl = Path("../output/stage1_result.jsonl")

model_id = "Qwen/Qwen2.5-VL-72B-Instruct"   
batch_size = 32

max_model_len = 8192
max_new_tokens = 512
temperature = 0.01
top_p = 1.0
repetition_penalty = 1.05
gpu_memory_utilization = 0.98
tensor_parallel_size = 2
limit_mm_per_prompt = {"image": 1}
dtype = "bfloat16"
enforce_eager = True
######## modify the above variables to fit your own situation ########

PROMPT = """Examine the given image and judge if it is a suitable panorama source.

**Validation criteria:**
- **format**: The image must be a 360° equirectangular panorama with Equirectangular Projection (ERP).
    - Mark **invalid** if:
        - The image looks like non-ERP panoramic format (e.g., flat perspective, dual fisheye, cube-map, cylindrical, little planet, or any projection other than ERP).
- **informative**: The image must contain clear, meaningful scene content without obstructions.
    - Mark **invalid** if:
        - The image contains **any** watermark, logo, text, or overlay, regardless of size or placement.
        - The image shows compression artifacts, stitching errors, severe motion blur, or pixelation.
        - The scene is too dark, obscured, or lacks visible detail (e.g., low-light/night scenes with little visibility).
        - The content is almost empty or uniform (e.g., mostly blank sky, solid color areas).
        - The image is rendered from a virtual environment.

**Instructions:**
- Give concise reasons for both judgments.
- A distortion is acceptable only if it follows ERP format.
- Be strict: if there is any doubt about ERP format or informativeness, mark **invalid**.
- Your response *MUST* follow the output schema in English.

**Output schema:**
{
    "format_reason": "<short reason within 20 words>",
    "format": "valid" | "invalid",
    "informative_reason": "<short reason within 20 words>",
    "informative": "valid" | "invalid"
}
"""

def build_llm_input(image_path: Path, prompt_text: str, processor: AutoProcessor):
    image_item = {
        "type": "image", "image": str(image_path),
        "min_pixels": 224 * 224,
        "max_pixels": 2160 * 28 * 28,
    }

    messages = [
        {"role": "system", "content": "You are a helpful panorama checking assistant."},
        {"role": "user", "content": [
            image_item,
            {"type": "text", "text": prompt_text},
        ]}
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    return {"prompt": prompt, "multi_modal_data": mm_data}


def prepare_one(obj, processor: AutoProcessor):
    path = data_root / obj["path"].split("_")[0] / obj["path"]
    prompt_text = PROMPT
    llm_in = build_llm_input(path, prompt_text, processor)
    return obj, llm_in


# -----------------------------
# load metadata
# -----------------------------

df = pd.read_csv(meta_file)
metadata = df["file_name"].tolist()
metadata = [{"path": path} for path in metadata]

for obj in tqdm(metadata, desc="checking files"):
    assert (data_root / obj["path"].split("_")[0] / obj["path"]).exists(), f"File not found: {data_root / obj['path'].split('_')[0] / obj['path']}"

# -----------------------------
# init vLLM + processor
# -----------------------------
llm = LLM(
    model=model_id,
    max_model_len=max_model_len,
    tensor_parallel_size=tensor_parallel_size,
    gpu_memory_utilization=gpu_memory_utilization,
    limit_mm_per_prompt=limit_mm_per_prompt,
    dtype=dtype,
    enforce_eager=enforce_eager,
)
processor = AutoProcessor.from_pretrained(model_id)
sampling_params = SamplingParams(
    max_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    repetition_penalty=repetition_penalty,
)

output_jsonl.parent.mkdir(parents=True, exist_ok=True)

max_workers = min(8, os.cpu_count() or 4)
inflight_limit = batch_size * 2
prepared_buffer = []

def infer_and_flush(buffer, writer):
    if not buffer:
        return
    batch_objs = [it[0] for it in buffer]
    batch_inputs = [it[1] for it in buffer]
    gens = llm.generate(batch_inputs, sampling_params)
    
    for ob, g in zip(batch_objs, gens):
        output = None
        try:
            output = parser(g.outputs[0].text)
            writer.write({
                "path": ob["path"],
                "format": output["format"],
                "format_reason": output["format_reason"],
                "informative": output["informative"],
                "informative_reason": output["informative_reason"]
            })
        except Exception as e:
            # write the null if parser does not contain the fields
            print(f"Error parsing output: {e}")
            temp = {"path": ob["path"], "format": "unknown", "format_reason": "null", "informative": "unknown", "informative_reason": "null"}
            for key in ["format", "format_reason", "informative", "informative_reason"]:
                if output and key in output:
                    temp[key] = output[key]
            writer.write(temp)

try:
    f = open(output_jsonl, "a", buffering=1, encoding="utf-8")
    writer = jsonlines.Writer(f)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        pbar = tqdm(total=len(metadata), desc="preparing & inferring")

        pending = set()
        i_submit = 0

        while i_submit < len(metadata) and len(pending) < inflight_limit:
            fut = ex.submit(prepare_one, metadata[i_submit], processor)
            pending.add(fut)
            i_submit += 1

        while pending:
            for fut in as_completed(list(pending), timeout=None):
                pending.remove(fut)
                try:
                    obj, llm_in = fut.result()
                    prepared_buffer.append((obj, llm_in))
                except Exception as e:
                    writer.write({
                        "path": obj["path"],
                        "format": "unknown",
                        "format_reason": "null",
                        "informative": "unknown",
                        "informative_reason": "null"
                    })

                pbar.update(1)

                if len(prepared_buffer) >= batch_size:
                    infer_and_flush(prepared_buffer[:batch_size], writer)
                    prepared_buffer = prepared_buffer[batch_size:]

                while i_submit < len(metadata) and len(pending) < inflight_limit:
                    fut_new = ex.submit(prepare_one, metadata[i_submit], processor)
                    pending.add(fut_new)
                    i_submit += 1
                break

        while prepared_buffer:
            chunk = prepared_buffer[:batch_size]
            infer_and_flush(chunk, writer)
            prepared_buffer = prepared_buffer[len(chunk):]

        pbar.close()
finally:
    try:
        writer.close()
    except Exception:
        pass
    try:
        f.close()
    except Exception:
        pass
    try:
        llm.shutdown()
    except Exception:
        pass

print(f"done. filter result saved to: {output_jsonl}")