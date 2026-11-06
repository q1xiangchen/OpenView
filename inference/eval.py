import argparse
import json
from pathlib import Path
import subprocess
import sys

from evaluation.eval_acc import parse_response


def summarize(predictions, rationale_results):
    judged = {item['id']: item for item in rationale_results}
    groups = {name: {'total': 0, 'choice_correct': 0, 'joint_correct': 0}
              for name in ('contextual', 'directional', 'overall')}
    missing = []
    for item in predictions:
        response = parse_response(item['response'])
        if response == 'Bad format':
            response = parse_response(item.get('response_answer'))
        choice_correct = item['answer'].lower() == response.lower()
        if choice_correct and item['id'] not in judged:
            missing.append(item['id'])
        joint_correct = choice_correct and bool(judged.get(item['id'], {}).get('is_correct', False))
        for name in ('overall', item['question_type']):
            group = groups[name]
            group['total'] += 1
            group['choice_correct'] += int(choice_correct)
            group['joint_correct'] += int(joint_correct)
    if missing:
        raise ValueError(f'Rationale evaluation incomplete: {len(missing)} correctly answered questions lack '
                         'judge results (failed API calls). Re-run the same command to retry them.')
    for group in groups.values():
        total, correct, joint = group['total'], group['choice_correct'], group['joint_correct']
        group['choice_accuracy'] = 100 * correct / total if total else None
        group['rationale_accuracy'] = 100 * joint / correct if correct else None
        group['joint_accuracy'] = 100 * joint / total if total else None
    return groups


def main():
    parser = argparse.ArgumentParser(description='Run benchmark evaluation and save summary metrics')
    parser.add_argument('--output_path', required=True, help='Path to prediction JSON')
    parser.add_argument('--model', default='gpt-5-mini', help='Judge model name')
    parser.add_argument('--base_url', default=None, help='OpenAI-compatible API base URL (default: OpenAI API)')
    parser.add_argument('--reasoning_effort', default='low', choices=['low', 'medium', 'high'],
                        help='Reasoning effort for GPT-5 / o-series judge models')
    parser.add_argument('--num_threads', type=int, default=8, help='Number of parallel judge requests')
    args = parser.parse_args()

    output_path = Path(args.output_path)
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    scripts = Path(__file__).parent / 'evaluation'
    subprocess.run([sys.executable, str(scripts / 'eval_acc.py'), '--output_path', str(output_path)], check=True)
    reasoning_cmd = [sys.executable, str(scripts / 'eval_reasoning.py'), '--output_path', str(output_path),
                     '--model', args.model, '--reasoning_effort', args.reasoning_effort,
                     '--num_threads', str(args.num_threads)]
    if args.base_url:
        reasoning_cmd += ['--base_url', args.base_url]
    subprocess.run(reasoning_cmd, check=True)

    result_dir = output_path.with_suffix('')
    with output_path.open() as f:
        predictions = json.load(f)
    with (result_dir / f'{args.model}_rationale_acc.json').open() as f:
        rationale_results = json.load(f)
    metrics = summarize(predictions, rationale_results)
    summary = {'judge_model': args.model, 'accuracy_unit': 'percent', 'metrics': metrics}
    summary_path = result_dir / f'{args.model}_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')

    lines = ['Group | Choice (%) | Rationale (%) | Joint (%)']
    for name, values in metrics.items():
        scores = [f'{values[key]:.2f}' if values[key] is not None else 'N/A'
                  for key in ('choice_accuracy', 'rationale_accuracy', 'joint_accuracy')]
        lines.append(f'{name} | ' + ' | '.join(scores))
    report = '\n'.join(lines) + '\n'
    print(report, end='')
    (result_dir / f'{args.model}_summary.txt').write_text(report)
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()
