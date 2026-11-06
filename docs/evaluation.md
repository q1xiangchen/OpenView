# Benchmark Evaluation

Generate predictions following [Benchmark Inference](inference.md), then run from the repository root:

```bash
cd inference
export OPENAI_API_KEY="your-key"
python eval.py \
  --output_path ../results/name_of_output.json \
  --model gpt-5-mini
```

This computes choice accuracy for all questions and LLM-judged rationale accuracy for the correctly answered ones. `--model` specifies the judge model. Results are saved under `results/name_of_output/`.

GPT-5 and o-series judges are called through the Responses API with `--reasoning_effort low` by default; other models use Chat Completions. To use another OpenAI-compatible judge (e.g. a local vLLM server), pass `--base_url`; the API key is read from `OPENAI_API_KEY`. Failed judge calls are not saved, so if the run reports missing judge results, re-run the same command to retry only those questions.

## Metrics

- **Choice accuracy:** the proportion of questions answered correctly.
- **Rationale accuracy:** the proportion of correctly answered questions whose rationales pass the judge evaluation.
- **Joint accuracy:** the proportion of all questions with both a correct answer and an accepted rationale.

