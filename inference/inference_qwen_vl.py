import os 
import json
import glob
import time
import torch
import argparse
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor

from utils.utils import extract_view, default_attn_implementation

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

# USER PROMPT TEMPLATE
USER_PROMPT_TEMPLATE = """
Question: {question}
Options: {options}
Instructions:
Analyze each option's correctness with reasoning or justification based on the image and the question.
Then, conclude with the single correct option in the format: <answer>A</answer>.
"""

class Qwen2_5VL:
    def __init__(
        self, model_path,
        max_new_tokens = 1024,
    ):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation=default_attn_implementation(),
            dtype="auto",
            device_map=device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.gen_config = {
            "max_new_tokens": max_new_tokens, 
            "use_cache": True
        }

    def parse_input(self, query=None, imgs=None, vid=None):
        content = []

        if imgs is not None:
            if isinstance(imgs, str):
                imgs = [imgs]
            for img in imgs:
                content.append({"type": "image", "image": img})

        if vid is not None:
            content.append({"type": "video", "video": vid})

        content.append({"type": "text", "text": query})
        return [{"role": "user", "content": content}]

    @torch.inference_mode()
    def chat(self, query=None, imgs=None, vid=None, history=None, get_logits=False):
        if history is None:
            history = [{"role": "system", "content": "You are a helpful assistant."}]

        history.extend(self.parse_input(query, imgs, vid))

        prompt_text = self.processor.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            history, return_video_kwargs=True
        )
        fps_inputs = video_kwargs.get("fps")

        proc_kwargs = dict(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if fps_inputs:
            proc_kwargs["fps"] = fps_inputs

        inputs = self.processor(**proc_kwargs).to(device)

        if get_logits:
            outputs = self.model.generate(
                **inputs, 
                **(self.gen_config | {"max_new_tokens": 1}), 
                return_dict_in_generate=True,
                output_scores=True
            )

            # return the token with the biggest logit
            token_id = outputs.scores[0].argmax(dim=-1).item()
            token_text = self.processor.batch_decode([[token_id]], skip_special_tokens=True)[0]
            return token_text
        else:
            outputs = self.model.generate(**inputs, **self.gen_config)

        trim = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = self.processor.batch_decode(
            trim, skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)[0]
        history.append({"role": "assistant", "content": response})
        
        return response, history

class Qwen3VL:
    def __init__(
        self, model_path,
        max_new_tokens = 1024,
    ):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation=default_attn_implementation(),
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.gen_config = {
            "max_new_tokens": max_new_tokens, 
            "use_cache": True
        }

    def parse_input(self, query=None, imgs=None, vid=None):
        content = []

        if imgs is not None:
            if isinstance(imgs, str):
                imgs = [imgs]
            for img in imgs:
                content.append({"type": "image", "image": img})

        if vid is not None:
            content.append({"type": "video", "video": vid})

        content.append({"type": "text", "text": query})
        return [{"role": "user", "content": content}]

    @torch.inference_mode()
    def chat(self, query=None, imgs=None, vid=None, history=None):
        history = self.parse_input(query, imgs, vid)

        prompt_text = self.processor.apply_chat_template(
            history,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = prompt_text.to(self.model.device)
        outputs = self.model.generate(**inputs, **self.gen_config)

        trim = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = self.processor.batch_decode(
            trim, skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)[0]
        
        return response, history


def merge_rank_files(output_json):
    """Merge all rank files into the main output file (called by rank 0 only)"""
    output_dir = os.path.dirname(output_json)
    output_basename = os.path.basename(output_json)
    output_name, output_ext = os.path.splitext(output_basename)
    
    # Find all rank files
    rank_pattern = os.path.join(output_dir, f"{output_name}_rank*{output_ext}")
    rank_files = sorted(glob.glob(rank_pattern))
    
    if not rank_files:
        return
    
    # Merge all results
    merged_results = []
    seen_ids = set()
    
    for rank_file in rank_files:
        if os.path.exists(rank_file):
            try:
                with open(rank_file, 'r') as f:
                    rank_data = json.load(f)
                for item in rank_data:
                    item_id = item.get("id")
                    if item_id not in seen_ids:
                        seen_ids.add(item_id)
                        merged_results.append(item)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Failed to parse rank file {rank_file}: {e}")
            except Exception as e:
                print(f"[WARNING] Error reading rank file {rank_file}: {e}")
    
    # read output json
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            output_data = json.load(f)
            for item in output_data:
                item_id = item.get("id")
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    merged_results.append(item)
    
    # Sort by ID for consistent ordering
    merged_results.sort(key=lambda x: x.get("id", ""))
    
    # Save merged results
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
        model_name = args.model_path.lower().replace(".", "_")
        if "qwen2_5" in model_name:
            model = Qwen2_5VL(model_path=args.model_path)
        elif "qwen3" in model_name:
            model = Qwen3VL(model_path=args.model_path)
        else:
            raise ValueError(f"Cannot infer Qwen version from model path '{args.model_path}'; "
                             "it must contain 'qwen2_5' / 'qwen2.5' or 'qwen3'.")

    for qa_pair in test_gt:
        data_id = qa_pair["id"]
        # Start timing this sample
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
        response, history = model.chat(query=query, imgs=[img])
        result["response"] = response

        if isinstance(model, Qwen2_5VL) and "3b" in args.model_path.lower():
            pred = model.chat(
                query="Give me only the final answer letter from your previous reply.",
                history=history, get_logits=True)
            result["response_answer"] = pred

        result_list.append(result)
        
        # End timing for this sample
        sample_time = time.time() - sample_start_time
        sample_times.append(sample_time)
        processed_count += 1

        # save the response to the rank-specific output json file
        with open(rank_output_json, "w") as f:
            json.dump(result_list, f, indent=4)
    
    # Calculate final statistics
    total_time = time.time() - total_start_time
    
    # Final merge
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
    parser = argparse.ArgumentParser(description="Video inference with Qwen-VL.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--test_gt_path", type=str, default="../annotations/OpenView_bench.json", help="Path to the test ground truth file.")
    parser.add_argument("--output_json", type=str, help="Path to the output JSON file.")
    parser.add_argument("--data_path", type=str, default="../data/test", help="Path to the Panorama Images.")
    args = parser.parse_args()
    
    if args.output_json is None:
        args.output_json = f"../results/{args.model_path.split('/')[-1]}_output.json"
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    main(args)