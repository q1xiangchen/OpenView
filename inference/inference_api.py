import os
import io
import base64
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

from utils.utils import extract_view

MAX_TRIALS = 3

THINKING_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

USER_PROMPT_TEMPLATE = """
Question: {question}
Options: {options}
Instructions:
Analyze each option's correctness with reasoning or justification based on the image and the question.
Then, conclude with the single correct option in the format: {box_format}."""

def encode_image(image):
    buffered = io.BytesIO()
    format_to_use = image.format if image.format else 'PNG'
    image.save(buffered, format=format_to_use)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class OpenAI_model:
    def __init__(self, api_key, base_url, model_name):
        self.model = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def chat(self, img=None, query=None, max_tokens=4096, temperature=0.0, top_p=0.95, history=None):
        tries = 0
        if history is None:
            messages = [
            {
                "role": "system", 
                "content": THINKING_SYSTEM_PROMPT if "InternVL" in self.model_name else "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img)}"}},
                ],
            }
        ]
        else:
            messages = history.copy()
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ],
            })

        while tries < MAX_TRIALS:
            if "Ovis2.5" in self.model_name:
                kwargs = dict(
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": True,
                        },
                        "mm_processor_kwargs": {
                            "images_kwargs": {
                                "min_pixels": 1048576,   # 1024 * 1024
                                "max_pixels": 3211264    # 1792 * 1792
                            }
                        }
                    }
                )
            elif "GLM-4.5V" in self.model_name:
                kwargs = dict(
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": True,
                        },
                    }
                )
            elif "InternVL" in self.model_name:
                kwargs = dict(
                    temperature=0.6,
                    top_p=0.95,
                )
            elif "gpt-5" in self.model_name:
                kwargs = dict(
                    max_completion_tokens=max_tokens,
                )
            elif "Qwen3-VL-8B-Instruct" in self.model_name:
                kwargs = dict(
                    max_tokens=max_tokens,
                    seed=3407,
                    temperature=0.7,
                    top_p=0.8,
                )
            elif "Qwen3-VL-8B-Thinking" in self.model_name:
                kwargs = dict(
                    max_tokens=max_tokens,
                    seed=1234,
                    temperature=0.6,
                    top_p=0.95,
                )
            elif "gemini" in self.model_name:
                kwargs = dict(
                    max_tokens=max_tokens,
                    reasoning_effort="low",
                )
            else:
                kwargs = dict(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                
            try:
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=messages, 
                    **kwargs
                )
                return response.choices[0].message.content, messages
            except Exception as e:
                print(f"Retrying... {tries + 1}/{MAX_TRIALS}; Error: {e}")
                # sleep for 1 second
                time.sleep(1)
                tries += 1

        print(f"Failed to get response after {MAX_TRIALS} tries")
        return None

class ThreadSafeResultHandler:
    def __init__(self, output_json):
        self.output_json = output_json
        self.results = []
        self.lock = threading.Lock()

        if os.path.exists(output_json):
            with open(output_json) as f:
                self.results = json.load(f)
        
    def add_result(self, result):
        with self.lock:
            self.results.append(result)
            # Save incrementally to avoid losing progress
            with open(self.output_json, "w") as f:
                json.dump(self.results, f, indent=4)

def process_qa_pair(args, model, qa_pair, result_handler, second_response=False):
    """Process a single Q&A pair in a thread"""
    qa_id = qa_pair["id"]
    
    try:
        pano_path = os.path.join(args.data_path, qa_pair["pano_name"])
        img = extract_view(pano_path, qa_pair)

        question = qa_pair["question"]
        options = [qa_pair["option_a"], qa_pair["option_b"], qa_pair["option_c"], qa_pair["option_d"], qa_pair["option_e"]]
        result = {"id": qa_id, "question": question, "options": options, "response": None, 
        "answer": qa_pair["answer"], "answer_reasoning": qa_pair["answer_reasoning"],
        "question_type": qa_pair["question_type"], "category": qa_pair["category"],
        "outdoor": qa_pair["outdoor"],
        }

        option_string = '\n'.join([letter + '. ' + option for letter, option in zip('ABCDE', options)])
        user_query = USER_PROMPT_TEMPLATE.format(
            question=question, options=option_string, 
            box_format="<answer>A</answer>",
            # box_format="\\boxed{A}" # change to this if the model cannot follow <answer> tags
        )
        response, history = model.chat(img=img, query=user_query)

        if second_response:
            second_response, _ = model.chat(
                query="Give me only the final answer letter from your previous reply.", 
                history=history
                )
            result["response_answer"] = second_response

        result["response"] = response
        result_handler.add_result(result)
        return True
        
    except Exception as e:
        print(f"Error processing qa_pair {qa_id}: {e}")
        return False

def main(args):
    model = OpenAI_model(api_key=args.api_key, base_url=args.base_url, model_name=args.model)
    result_handler = ThreadSafeResultHandler(args.output_json)
    second_response = True if "GLM" in args.model else False

    # skip the qa_pairs that are already in the json file
    existings = set()
    if os.path.exists(args.output_json):
        with open(args.output_json) as f:
            gt_data = json.load(f)
            for item in gt_data:
                existings.add(item["id"])

    # read the data list
    with open(args.test_gt_path, "r") as f:
        test_gt = json.load(f)

    # Filter out already processed items
    remaining_qa_pairs = [qa_pair for qa_pair in test_gt if qa_pair["id"] not in existings]
    
    if not remaining_qa_pairs:
        print("All Q&A pairs have already been processed!")
        return

    print(f"Processing {len(remaining_qa_pairs)} remaining Q&A pairs with {args.num_threads} threads...")

    # Process remaining items in parallel
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all tasks
        future_to_qa = {
            executor.submit(process_qa_pair, args, model, qa_pair, result_handler, second_response): qa_pair 
            for qa_pair in remaining_qa_pairs
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_qa), total=len(remaining_qa_pairs), desc="Processing Q&A pairs"):
            qa_pair = future_to_qa[future]
            
            try:
                future.result()
            except Exception as e:
                print(f"Unexpected error processing {qa_pair['id']}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_gt_path", type=str, required=True, help="Path to the test ground truth file.")
    parser.add_argument("--base_url", type=str, help="Base URL for the API.")
    parser.add_argument("--model", type=str, required=True, help="Model name.")
    parser.add_argument("--output_json", type=str, help="Path to the output JSON file.")
    parser.add_argument("--data_path", type=str, default="../data/test", help="Path to the Panorama Images.")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads for parallel processing.")
    args = parser.parse_args()

    if "gpt" in args.model:
        args.api_key = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY") ## Modify this
        args.base_url = "https://api.openai.com/v1"
    elif "gemini" in args.model:
        args.api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY") ## Modify this
        args.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    else:
        # use vLLM server
        args.api_key = "EMPTY"


    if args.output_json is None:
        args.output_json = f"../results/{args.model.split('/')[-1]}_output.json"
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    main(args)