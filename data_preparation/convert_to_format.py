import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils import extract_view

def process_item(item):
    """Process a single item in a worker process"""
    try:
        # extract the question image
        image_path = extract_view(item, save_dir=f"../data/{suffix}")
        patch_images = [image_path]

        options = [item["option_" + letter] for letter in "abcde"]
        option_string = '\n'.join([letter + '. ' + option for letter, option in zip('ABCDE', options)])
        image_tokens = "<image>\n" * len(patch_images)
        qa_obj = {
            "id": item['id'],
            "image": patch_images,
            "conversations": [{
                "from": "human",
                "value": (
                    f"{image_tokens}"
                    f"Question: {item['question']}\nOptions: {option_string}\nInstructions:\nAnalyze each option's correctness with reasoning or justification based on the image and the question.\nThen, conclude with the single correct option in the format: <answer>YOUR_ANSWER_HERE</answer>."
                    )
            }, {
                "from": "gpt",
                "value": f"{item['answer_reasoning']}\n<answer>{item['answer']}</answer>"
            }]
        }
        return qa_obj
    except Exception as e:
        print(f"Error processing item {item.get('id', 'unknown')}: {e}")
        return None

def main(args):
    global suffix
    suffix = args.input_file.split("/")[-1].split(".json")[0]

    # Load the original JSON file
    with open(args.input_file, "r") as f:
        raw_data = json.load(f)

    # Shuffle with fixed seed
    random.seed(42)
    random.shuffle(raw_data)

    # Determine number of processes
    num_processes = cpu_count()
    print(f"Processing {len(raw_data)} items using {num_processes} processes...")

    # Convert format using multiprocessing
    converted = []
    with Pool(processes=num_processes) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_item, raw_data, chunksize=1),
            total=len(raw_data),
            desc="Converting items"
        ))
        
        # Filter out None results (failed items)
        converted = [result for result in results if result is not None]

    # Save to output JSON
    with open(args.output_file, "w") as fout:
        json.dump(converted, fout, indent=2)

    print(f"Converted {len(converted)} QA entries to '{args.output_file}'")
    print(f"Successfully processed {len(converted)}/{len(raw_data)} items")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, help="Path to output JSON file (default: <input-stem>_converted.json beside the input)")
    args = parser.parse_args()

    if args.output_file is None:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.with_name(f"{input_path.stem}_converted.json"))
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    main(args)
