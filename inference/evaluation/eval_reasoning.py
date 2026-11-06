import json
import time
import argparse
import os
from tqdm import tqdm
from openai import OpenAI
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

from eval_acc import parse_response


MAX_TRIES = 3

USER_PROMPT_TEMPLATE = """You are a rigorous evaluator for an out-of-view (OOV) multi-choice visual question answering task.

Your job is to determine whether the model's response is logically consistent with the ground truth and factual correct. Notice that the image is not provided for this evaluation. Evaluate both the option-level rationale and the final answer justification holistically.

Case of INCORRECT:
1. Any answers based primarily on guessing, imagination or scene stereotypes without referring to image evidence.
2. Any rationale that only say "not visible", "unlikely", "probably", or other vague statements without supporting evidence.
3. Missing the rationale for any five options.
4. Responses that omit essential evidence, or only provide the final choice should be considered incorrect.

Important reminders:
- Long explanations do NOT necessarily mean correct reasoning.
- Be strict with each criterion.

Question: {question}

Options: {options}

Ground Truth Answer: {answer_reasoning} {answer}

Response: {response}

Respond ONLY with one of the following:

<answer>Yes</answer>
or
<answer>No</answer>
"""


def is_reasoning_model(model_name: str) -> bool:
    """OpenAI reasoning models (GPT-5, o-series) need the Responses API with reasoning effort."""
    name = model_name.lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAI_model:
    def __init__(self, api_key, base_url, model_name, reasoning_effort="low"):
        self.model = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

    def chat(self, query: str) -> Optional[str]:
        """Return the judge's text output, or None if every attempt failed."""
        if is_reasoning_model(self.model_name):
            return self._chat_reasoning(query)
        return self._chat_completion(query)

    def _chat_completion(self, query, max_tokens=128, temperature=0.0, top_p=0.95):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
        for trial in range(MAX_TRIES):
            try:
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Retrying... {trial + 1}/{MAX_TRIES}; Error: {e}")
                time.sleep(2 ** trial)
        return None

    def _chat_reasoning(self, query, max_output_tokens=2048):
        input_items = [{"role": "user", "content": [{"type": "input_text", "text": query}]}]
        for trial in range(MAX_TRIES):
            try:
                response = self.model.responses.create(
                    model=self.model_name,
                    instructions="You are a helpful assistant.",
                    input=input_items,
                    reasoning={"effort": self.reasoning_effort},
                    max_output_tokens=max_output_tokens,
                )
                details = getattr(response, "incomplete_details", None)
                if (response.status == "incomplete" and not response.output_text
                        and getattr(details, "reason", None) == "max_output_tokens"):
                    print(f"Retrying... {trial + 1}/{MAX_TRIES}; Ran out of tokens during reasoning")
                    max_output_tokens = min(max_output_tokens + 4096, 128000)
                    continue
                return response.output_text or ""
            except Exception as e:
                print(f"Retrying... {trial + 1}/{MAX_TRIES}; Error: {e}")
                time.sleep(2 ** trial)
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


def evaluate_single_item(item: Dict, model: OpenAI_model) -> Tuple[str, Optional[str], bool]:
    """
    Evaluate a single item and return (qa_id, judge_response, is_correct).
    judge_response is None if the API call failed.
    """
    qa_id = item["id"]
    option_string = '\n'.join([letter + '. ' + option for letter, option in zip('ABCDE', item["options"])])
    response = item["response"] or ""

    if "<think>" in response and "</think>" in response:
        # skip the thinking part
        response = "".join(response.split("</think>")[1:]).strip()

    if len(response) < 30:
        return qa_id, "Skipped call", False

    user_query = USER_PROMPT_TEMPLATE.format(
        question=item["question"],
        options=option_string,
        answer_reasoning=item["answer_reasoning"],
        answer=f"<answer>{item['answer']}</answer>",
        response=response,
    )

    api_response = model.chat(query=user_query)
    if api_response is None:
        return qa_id, None, False

    try:
        answer = api_response.split("<answer>")[1].split("</answer>")[0].strip()
    except IndexError:
        answer = "Bad format"

    is_correct = answer.lower() == "yes"
    return qa_id, api_response, is_correct


def process_eval_item(model, item, result_handler):
    """Process a single evaluation item in a thread. Returns False if the judge call failed."""
    qa_id, api_response, is_correct = evaluate_single_item(item, model)
    if api_response is None:
        # Do not save failed calls, so that re-running the script retries them.
        return False

    result = {
        "id": qa_id,
        "category": item.get("category", None),
        "outdoor": item.get("outdoor", None),
        "question_type": item.get("question_type", None),
        "question": item["question"],
        "options": item["options"],
        "answer_reasoning": item["answer_reasoning"],
        "response_reasoning": item["response"],
        "eval_response": api_response,
        "is_correct": is_correct
    }

    result_handler.add_result(result)
    return True


def percent(numerator, denominator):
    return f"{numerator / denominator * 100}%" if denominator else "N/A"


def report(eval_data, report_path=None):
    """Print (and optionally save) rationale accuracy over the judged items."""
    stats = {}
    for name in ("directional", "contextual"):
        items = [item for item in eval_data if item["question_type"] == name]
        stats[name] = (sum(bool(item.get("is_correct")) for item in items), len(items))
    acc = sum(bool(item.get("is_correct")) for item in eval_data)
    total = len(eval_data)

    lines = [
        f"Total: {total}",
        f"Directional Reasoning Accuracy: {percent(*stats['directional'])}",
        f"Contextual Reasoning Accuracy: {percent(*stats['contextual'])}",
        f"Reasoning Accuracy: {percent(acc, total)}",
    ]
    print("\n".join(lines))
    if report_path is not None:
        with open(report_path, "w") as f:
            f.write("\n".join(lines) + "\n")


def main(args):
    output_path = args.output_path
    eval_output_path = args.eval_output_json

    with open(output_path, "r") as f:
        data = json.load(f)

    result_handler = ThreadSafeResultHandler(eval_output_path)
    existings = {item["id"] for item in result_handler.results}

    # Only correctly answered questions are judged; skip ones already judged.
    choice_correct_ids = set()
    remaining_items = []
    for item in data:
        response = parse_response(item["response"])
        if response == "Bad format":
            response = parse_response(item.get("response_answer", None))
        if item["answer"].lower() != response.lower():
            continue
        choice_correct_ids.add(item["id"])
        if item["id"] not in existings:
            remaining_items.append(item)

    def current_results():
        # Ignore stale judge entries for questions that are no longer answered correctly.
        return [r for r in result_handler.results if r["id"] in choice_correct_ids]

    report_path = output_path[:-5] + f"/{args.model}_reasoning_eval.txt"
    if not remaining_items:
        print("All items have already been processed!")
        report(current_results(), report_path)
        return

    print(f"Processing {len(remaining_items)} remaining items with {args.num_threads} threads...")
    model = OpenAI_model(api_key=args.api_key, base_url=args.base_url,
                         model_name=args.model, reasoning_effort=args.reasoning_effort)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        future_to_item = {executor.submit(process_eval_item, model, item, result_handler): item
                          for item in remaining_items}
        for future in tqdm(as_completed(future_to_item), total=len(remaining_items), desc="Evaluating"):
            item = future_to_item[future]
            try:
                if not future.result():
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"Unexpected error processing {item['id']}: {e}")

    report(current_results(), report_path)
    if failed:
        print(f"WARNING: {failed} judge calls failed and were not saved. Re-run the same command to retry them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--eval_output_json", type=str, help="Path to the evaluation output JSON file.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Judge model name")
    parser.add_argument("--base_url", type=str, default=None,
                        help="OpenAI-compatible API base URL (default: OpenAI API)")
    parser.add_argument("--reasoning_effort", type=str, default="low", choices=["low", "medium", "high"],
                        help="Reasoning effort for GPT-5 / o-series judge models")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads for parallel processing")
    args = parser.parse_args()

    # The API key is read from OPENAI_API_KEY (any non-empty value works for local servers).
    args.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    if args.eval_output_json is None:
        args.eval_output_json = args.output_path[:-5] + f"/{args.model}_rationale_acc.json"
    os.makedirs(os.path.dirname(args.eval_output_json), exist_ok=True)

    main(args)
