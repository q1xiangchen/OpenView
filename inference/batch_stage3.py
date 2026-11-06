from utils.vllm_agent import build_vllm_agent

import os, yaml, argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.utils import read_variable_saves, save_variable


def load_config(path):
    with open(path, "r") as f: 
        return yaml.safe_load(f)

def process_pano(pano_info, cfg, backend, question_type, overwrite=False):
    # Stage 3: proposing QA pairs
    pano_path, _ = pano_info

    # get pano info
    pano_id = Path(pano_path).stem
    patch_prefix = f"{cfg['patch_cols']}x{cfg['patch_rows']}"
    
    # overwrite with cache directory if exists
    cache_dir = Path(cfg["cache_dir"]) / pano_id / patch_prefix / backend
    out_dir = Path(cfg["out_dir"]) / pano_id / patch_prefix / backend / question_type
    os.makedirs(out_dir, exist_ok=True)

    # read or generate proposals
    try:
        proposals = read_variable_saves(out_dir, "proposals")
    except Exception:
        proposals = None

    if proposals and not overwrite: 
        print(f"[INFO] Proposals loaded from cache: {out_dir}")
        return None

    captions = read_variable_saves(cache_dir, "captions")
    summary = read_variable_saves(cache_dir, "summary")

    if not captions or not summary:
        print(f"[INFO] Captions or summary not found: {cache_dir}")
        return None

    return captions, summary, pano_path, out_dir

def infer_and_flush(buffer, agent, k, question_type):
    if not buffer:
        return
    batch_captions  = [it[0] for it in buffer]
    batch_summary   = [it[1] for it in buffer]
    batch_pano_path = [it[2] for it in buffer]
    batch_out_dir   = [it[3] for it in buffer]

    # proposing
    batch_result = agent.propose(
        batch_summary, batch_captions, 
        batch_pano_path, k, question_type
        )

    for idx, result in enumerate(batch_result):
        save_variable(result, batch_out_dir[idx], "proposals")


def main(args):
    # load cfg
    cfg = load_config(path=args.config_path)
    data_dir = cfg["data_dir"]
    out_dir = cfg["out_dir"]
    question_types = cfg["question_types"]
    k = cfg["proposals_per_generator"]
    os.makedirs(out_dir, exist_ok=True)
    overwrite = args.overwrite

    print(f"[INFO] Config: {cfg}")

    # prepare all files
    df = pd.read_csv(cfg["annotation_path"])
    pano_files = df.itertuples(index=False, name=None)
    pano_files = [(os.path.join(data_dir, o[0].split("_")[0], o[0]), o[1]) for o in pano_files]

    print(f"[INFO] Find {len(pano_files)} panoramas to process")

    # init agent
    agent = build_vllm_agent(cfg)

    # prepare the batch
    max_workers = min(8, os.cpu_count() or 4)
    batch_size = cfg["batch_size"]
    inflight_limit = batch_size * 2    

    for question_type in question_types:
        print(f"[INFO] Processing question type: {question_type}...")
        prepared_buffer = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            pbar = tqdm(total=len(pano_files), desc="Processing panoramas")

            pending = set()
            i_submit = 0

            while i_submit < len(pano_files) and len(pending) < inflight_limit:
                fut = ex.submit(process_pano, pano_files[i_submit], cfg, agent.backend, question_type, overwrite)
                pending.add(fut)
                i_submit += 1
            
            while pending:
                for fut in as_completed(list(pending), timeout=None):
                    pending.remove(fut)
                    try:
                        returns = fut.result()
                        if returns:
                            prepared_buffer.append(returns)
                    except Exception as e:
                        print(f"[ERROR] Error processing panorama: {e}")

                    pbar.update(1)

                    if len(prepared_buffer) >= batch_size:
                        infer_and_flush(prepared_buffer[:batch_size], agent, k, question_type)
                        prepared_buffer = prepared_buffer[batch_size:]

                    while i_submit < len(pano_files) and len(pending) < inflight_limit:
                        fut_new = ex.submit(process_pano, pano_files[i_submit], cfg, agent.backend, question_type, overwrite)
                        pending.add(fut_new)
                        i_submit += 1

                    break

            while prepared_buffer:
                chunk = prepared_buffer[:batch_size]
                infer_and_flush(chunk, agent, k, question_type)
                prepared_buffer = prepared_buffer[len(chunk):]

            pbar.close()
    agent.model.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/batch_config.yaml")
    parser.add_argument("--overwrite", "-o", action="store_true")
    args = parser.parse_args()
    main(args)