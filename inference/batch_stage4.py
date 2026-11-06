import os, yaml, argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import re
import random
from utils.utils import read_variable_saves, spec_check

LETTER_SINGLE_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", flags=re.IGNORECASE)
ANY_LETTER_MENTION_RE = re.compile(r"(?<![A-Za-z])([ABCDE])(?![A-Za-z])", flags=re.IGNORECASE)
OPTION_WORD_MENTION_RE = re.compile(r"\boption\s+[abcde]\b", flags=re.IGNORECASE)


def load_config(path):
    with open(path, "r") as f: 
        return yaml.safe_load(f)

def process_pano(pano_path, cfg, question_type):
    # Stage 4: proposal refiner
    # get pano info
    pano_id = Path(pano_path).stem
    patch_prefix = f"{cfg['patch_cols']}x{cfg['patch_rows']}"
    backend = cfg["agent_config"]["model_id"].split("/")[-1]
    
    # overwrite with cache directory if exists
    cache_dir = Path(cfg["cache_dir"]) / pano_id / patch_prefix / backend
    out_dir = Path(cfg["out_dir"]) / pano_id / patch_prefix / backend / question_type

    # read or generate proposals
    proposals = read_variable_saves(out_dir, "proposals")
    if proposals: 
        sample = proposals[0]
        if "outdoor" not in sample or sample["outdoor"] == "Not Given":
            summary = read_variable_saves(cache_dir, "summary")
            if summary:
                for proposal in proposals:
                    proposal["outdoor"] = summary.outdoor
        return proposals[:3]
    else:
        return False

def main(args):
    # load cfg
    cfg = load_config(path=args.config_path)
    data_dir = cfg["data_dir"]
    question_types = cfg["question_types"]
    save_path = cfg.get("save_path", "../output/OpenView_generated_vqa.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[INFO] Config: {cfg}")

    # prepare all files
    df = pd.read_csv(cfg["annotation_path"])
    pano_files = df.itertuples(index=False, name=None)
    pano_files = [os.path.join(data_dir, o[0].split("_")[0], o[0]) for o in pano_files]

    print(f"[INFO] Find {len(pano_files)} panoramas to process")

    proposal_list = []
    qa_id = 0
    shuffle_number = 0
    low_confidence = 0
    invalid_spec = 0
    for question_type in question_types:
        acc = 0
        for pano_file in tqdm(pano_files, desc="Processing panoramas"):
            proposals = process_pano(pano_file, cfg, question_type)
            if not proposals: continue
            acc += 1

            # get all useful attributes
            for item in proposals:
                shuffle_combo = [False, True]
                for shuffle in shuffle_combo:
                    proposal = item.copy()
                    # pass the non valid proposal
                    if not spec_check(proposal): 
                        invalid_spec += 1
                        continue
                    if int(proposal['confidence_score']) < 3: 
                        low_confidence += 1
                        continue

                    # check if answer_reasoning contains any option A-E
                    ans_reason = (proposal.get("answer_reasoning") or "")
                    if ANY_LETTER_MENTION_RE.search(ans_reason):
                        proposal["answer_reasoning"] = ""

                    # check if any option A-D in each reasoning then dont shuffle
                    reasonings = [
                        proposal.get("option_a_reasoning") or "",
                        proposal.get("option_b_reasoning") or "",
                        proposal.get("option_c_reasoning") or "",
                        proposal.get("option_d_reasoning") or "",
                    ]
                    if any(OPTION_WORD_MENTION_RE.search(rz) for rz in reasonings):
                        shuffle = False

                    option_tuples = [
                        ("A", proposal.get("option_a", ""), proposal.get("option_a_reasoning", "")),
                        ("B", proposal.get("option_b", ""), proposal.get("option_b_reasoning", "")),
                        ("C", proposal.get("option_c", ""), proposal.get("option_c_reasoning", "")),
                        ("D", proposal.get("option_d", ""), proposal.get("option_d_reasoning", "")),
                    ]

                    if shuffle:
                        shuffle_number += 1

                        # Creat a random shuffled order
                        new_order = random.sample(range(4), 4)
                        new_options = [option_tuples[i][1] for i in new_order]
                        new_option_reasons = [option_tuples[i][2] for i in new_order]
                        orig_answer_letter = (proposal.get("answer") or "").strip().upper()

                        # Build mapping from old letter to new letter
                        old_to_new_letter = {
                            option_tuples[i][0]: "ABCD"[new_order.index(i)] for i in range(4)
                        }

                        def remap_letters(text: str) -> str:
                            if not text:
                                return text
                            def repl(m):
                                old = m.group(1).upper()
                                # preserve case of match (optional):
                                new_letter = old_to_new_letter.get(old, old)
                                return new_letter
                            return LETTER_SINGLE_RE.sub(repl, text)

                        # Option E: keep position, but remap any A–D references it contains
                        proposal["option_e"] = remap_letters(proposal.get("option_e", ""))
                        proposal["option_e_reasoning"] = remap_letters(proposal.get("option_e_reasoning", ""))
                        
                        # Only shuffle A-D, E means "None of the above" and is always unshuffled
                        if orig_answer_letter in "ABCD":
                            # Find index of the original answer
                            orig_answer_idx = ord(orig_answer_letter) - ord("A")
                            new_answer_idx = new_order.index(orig_answer_idx)
                            new_answer = "ABCD"[new_answer_idx]
                        elif orig_answer_letter == "E":
                            new_answer = "E"
                        else:
                            new_answer = orig_answer_letter
                    else:
                        # no shuffle
                        new_options = [
                            proposal.get("option_a", ""),
                            proposal.get("option_b", ""),
                            proposal.get("option_c", ""),
                            proposal.get("option_d", ""),
                        ]
                        new_option_reasons = [
                            proposal.get("option_a_reasoning", ""),
                            proposal.get("option_b_reasoning", ""),
                            proposal.get("option_c_reasoning", ""),
                            proposal.get("option_d_reasoning", ""),
                        ]
                        new_answer = (proposal.get("answer") or "").strip().upper()
                    
                    # Build the answer_reasoning using the new option mapping
                    final_reasoning = ""
                    rationale_prefixs = [
                        ("Option ", " rationale:"),
                        ("- ", "."),
                        ("**", "**")
                    ]
                    rationale_prefix = random.choice(rationale_prefixs)
                    for i, letter in enumerate("ABCD"):
                        final_reasoning += f"{rationale_prefix[0]}{letter}{rationale_prefix[1]} {new_option_reasons[i]}\n"
                    final_reasoning += f"{rationale_prefix[0]}E{rationale_prefix[1]} {proposal['option_e_reasoning']}"
                    conclusions = ["In conclusion, ", "To summarize, ", "In summary, ", "To conclude, "]
                    if proposal.get("answer_reasoning"):
                        final_reasoning += "\n" + random.choice(conclusions) + proposal["answer_reasoning"]

                    u_offset = random.choice([-1, 1]) * random.uniform(0, 0.005)
                    v_offset = random.choice([-1, 1]) * random.uniform(0, 0.01)
                    proposal["u_norm"] = round(float(proposal["u_norm"]) + u_offset, 4)
                    proposal["v_norm"] = round(float(proposal["v_norm"]) + v_offset, 4)

                    proposal["u_norm"] = proposal["u_norm"] % 1
                    proposal["v_norm"] = proposal["v_norm"] % 1
                    
                    proposal_list.append({
                        "id": qa_id,
                        # view meta
                        "u_norm": proposal["u_norm"],
                        "v_norm": proposal["v_norm"],
                        "diag_fov": proposal["diag_fov"],
                        "image_size": proposal["image_size"],
                        # pano meta
                        "pano_path": proposal["pano_path"],
                        "category": proposal["category"],
                        "outdoor": proposal["outdoor"],
                        # question meta
                        "question_type": proposal["question_type"],
                        "question": proposal["question"],
                        "option_a": new_options[0],
                        "option_b": new_options[1],
                        "option_c": new_options[2],
                        "option_d": new_options[3],
                        "option_e": proposal["option_e"],
                        "answer": new_answer,
                        "answer_reasoning": final_reasoning,
                    })
                    qa_id += 1
        print(f"[INFO] Found {acc}/{len(pano_files)} pano with proposals for question type: {question_type}")
    print(f"[INFO] In total, {len(proposal_list)} QA pairs")
    print(f"[INFO] {low_confidence//len(shuffle_combo)} QA pairs with confidence score less than 3")
    print(f"[INFO] {invalid_spec//len(shuffle_combo)} QA pairs with invalid spec")
    print(f"[INFO] Shuffled {shuffle_number} QA pairs")
    print(f"[INFO] Saved to {save_path}")

    with open(save_path, "w") as f:
        json.dump(proposal_list, f, indent=4)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/batch_config.yaml")
    args = parser.parse_args()
    main(args)