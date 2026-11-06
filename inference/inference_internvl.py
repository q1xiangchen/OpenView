import os 
import torch
import torchvision.transforms as T
import argparse
import time
import glob
import json

from utils.utils import extract_view, default_attn_implementation
from transformers import AutoProcessor, AutoModelForImageTextToText

def get_rank_and_world():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    else:
        rank = int(os.getenv("SHARD", 0))

    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_NTASKS" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        world_size = int(os.getenv("NSHARDS", 1))

    if "LOCAL_RANK" in os.environ:
        device_id = int(os.environ["LOCAL_RANK"])
    else:
        device_id = 0

    return rank, world_size, device_id

rank, world_size, device_id = get_rank_and_world()
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

if os.getenv("USE_DDP", "0") == "1":
    torch.distributed.init_process_group("nccl")

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

# USER PROMPT TEMPLATE
USER_PROMPT_TEMPLATE = """<image>
Question: {question}
Options: {options}
Instructions:
Analyze each option's correctness with reasoning or justification based on the image and the question.
Then, conclude with the single correct option in the format: <answer>A</answer>.
"""

class InternVL:
    def __init__(
        self, model_path,
        max_new_tokens = 1024,
    ):
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            attn_implementation=default_attn_implementation(),
            trust_remote_code=True,
            device_map="auto"
        ).eval()

        self.gen_config = dict(max_new_tokens=max_new_tokens, do_sample=True)
    
    @torch.inference_mode()
    def inference(self, query=None, imgs=None):
        assert imgs and imgs[0] is not None, "imgs[0] (PIL.Image) cannot be empty"
        image = imgs[0]

        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image"}
            ]},
        ]

        try:
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        except Exception:
            text = query

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        outputs = self.model.generate(**inputs, **self.gen_config)

        trim = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], outputs)
        ]
        response = self.processor.batch_decode(
            trim,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response


def merge_rank_files(output_json):
    """Merge all rank files into the main output file (called by rank 0 only)"""
    output_dir = os.path.dirname(output_json)
    output_basename = os.path.basename(output_json)
    output_name, output_ext = os.path.splitext(output_basename)
    
    # Find all rank files
    rank_pattern = os.path.join(output_dir, f"{output_name}_rank*{output_ext}")
    rank_files = sorted(glob.glob(rank_pattern))
    
    if not rank_files and not os.path.exists(output_json):
        return
    
    # Merge all results
    merged_results = []
    seen_ids = set()
    
    def _ingest(path):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                for item in data:
                    iid = item.get("id")
                    if iid not in seen_ids:
                        seen_ids.add(iid)
                        merged_results.append(item)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Failed to parse file {path}: {e}")
            except Exception as e:
                print(f"[WARNING] Error reading file {path}: {e}")

    for rf in rank_files:
        _ingest(rf)
    _ingest(output_json)

    merged_results.sort(key=lambda x: x.get("id", ""))
    
    with open(output_json, 'w') as f:
        json.dump(merged_results, f, indent=4)

def main(args):
    merge_rank_files(args.output_json)
    print(f"[INFO] Inference with model: {args.model_path}")
    result_list = []

    # Timing statistics
    sample_times = []
    total_start_time = time.time()

    # Create rank-specific output file
    output_dir = os.path.dirname(args.output_json)
    output_basename = os.path.basename(args.output_json)
    output_name, output_ext = os.path.splitext(output_basename)
    rank_output_json = os.path.join(output_dir, f"{output_name}_rank{rank}{output_ext}")
    
    print(f"[INFO] Rank {rank} will write to: {rank_output_json}")

    # Check existing results from the MERGED file (not rank-specific)
    existings = set()
    if os.path.exists(args.output_json):
        with open(args.output_json) as f:
            merged_data = json.load(f)
            for item in merged_data:
                existings.add(item["id"])
        print(f"[INFO] Rank {rank} found {len(existings)} existing results")
    
    # Also load rank-specific file to resume this rank's work
    if os.path.exists(rank_output_json):
        with open(rank_output_json) as f:
            gt_data = json.load(f)
            for item in gt_data:
                result_list.append(item)
                existings.add(item["id"])
        print(f"[INFO] Rank {rank} loaded {len(gt_data)} items from rank-specific file")

    # read the data list
    def shard_indices(n, rank, world_size):
        base = n // world_size
        extra = n % world_size
        start = rank * base + min(rank, extra)
        end = start + base + (1 if rank < extra else 0)
        return start, end
        
    with open(args.test_gt_path, "r") as f:
        test_gt = json.load(f)
        test_gt = [item for item in test_gt if item["id"] not in existings]
    print(f"[INFO] Total {len(test_gt)} items")

    s, e = shard_indices(len(test_gt), rank, world_size)
    test_gt = test_gt[s:e]
    print(f"[INFO] Rank {rank} found {len(test_gt)} items to process")

    processed_count = 0
    if len(test_gt) > 0:
        model = InternVL(model_path=args.model_path)

    for qa_pair in test_gt:
        data_id = qa_pair["id"]
        sample_start_time = time.time()
        
        pano_path = qa_pair.get("pano_path", os.path.join(args.data_path, qa_pair.get("pano_name", "")))
        img = extract_view(pano_path, qa_pair)

        question = qa_pair["question"]
        options = [qa_pair["option_a"], qa_pair["option_b"], qa_pair["option_c"], qa_pair["option_d"], qa_pair["option_e"]]
        result = {
            "id": data_id, "question": question, "options": options, 
            "response": None, "response_answer": None,
            "answer": qa_pair["answer"], "answer_reasoning": qa_pair["answer_reasoning"],
            "question_type": qa_pair["question_type"], "category": qa_pair["category"],
            "outdoor": qa_pair["outdoor"],
            }

        option_string = '\n'.join([letter + '. ' + option for letter, option in zip('ABCDE', options)])
        query = USER_PROMPT_TEMPLATE.format(question=question, options=option_string)
        response = model.inference(query=query, imgs=[img])
        result["response"] = response
        result_list.append(result)
        
        sample_time = time.time() - sample_start_time
        sample_times.append(sample_time)
        processed_count += 1

        with open(rank_output_json, "w") as f:
            json.dump(result_list, f, indent=4)
    
    total_time = time.time() - total_start_time
    
    merge_rank_files(args.output_json)
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[INFO] Rank {rank} FINISHED")
        print(f"[INFO] Total items processed: {processed_count}")
        print(f"[INFO] Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        if processed_count > 0:
            avg_time = sum(sample_times) / len(sample_times)
            print(f"[INFO] Average time per sample: {avg_time:.2f}s")
            print(f"[INFO] Throughput: {processed_count/total_time:.4f} samples/second")
        print(f"[INFO] Rank {rank} output saved to: {rank_output_json}")
        print(f"[INFO] Final merged output saved to: {args.output_json}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"[INFO] Rank {rank} FINISHED")
        print(f"[INFO] Total items processed: {processed_count}")
        print(f"[INFO] Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        if processed_count > 0:
            avg_time = sum(sample_times) / len(sample_times)
            print(f"[INFO] Average time per sample: {avg_time:.2f}s")
            print(f"[INFO] Throughput: {processed_count/total_time:.4f} samples/second")
        print(f"[INFO] Rank {rank} output saved to: {rank_output_json}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video inference with InternVL.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--test_gt_path", type=str, default="../annotations/OpenView_bench.json", help="Path to the test ground truth file.")
    parser.add_argument("--output_json", type=str, help="Path to the output JSON file.")
    parser.add_argument("--data_path", type=str, default="../data/test", help="Path to the Panorama Images.")
    args = parser.parse_args()
    
    if args.output_json is None:
        args.output_json = f"../results/{args.model_path.split('/')[-1]}_output.json"
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    main(args)