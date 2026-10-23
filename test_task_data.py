"""Dataset/scoring regressions and CPU integration of the mixed-domain pipeline."""
import json
from pathlib import Path

import pytest

from common import extract_answer, format_instruction, normalize_math, score_tokens, write_json
from task_data import build_splits, dataset_specs, load_pools, normalize_row, question_hash


def qa(question, gold='B', labels=None):
    labels = labels or ['A', 'B', 'C']
    return dict(id=question, question=question, choices=dict(label=labels, text=['option '+x for x in labels]), answerKey=gold)


def sky(question, difficulty=2, gold=None):
    return dict(prompt=[dict(role='user', content=question)], reward_model=dict(ground_truth=json.dumps(gold or ['42'])),
                extra_info=dict(model_difficulty={'DeepSeek-R1-Distill-Qwen-1.5B': difficulty}))


def corpus(root):
    specs = [dict(name=name, train_n=3, val_n=2, test_n=2)
             for name in ['gsm8k', 'skywork_easy', 'commonsense_qa', 'arc_easy']]
    sources = {
        'gsm8k_train': [dict(question=f'g train {i}', answer=f'<<1+1=2>>\n#### {i}') for i in range(14)],
        'gsm8k_test': [dict(question=f'g test {i}', answer=f'#### {i}') for i in range(8)],
        'skywork_easy_math': [sky(f'sky {i}') for i in range(14)],
        'commonsense_qa_train': [qa(f'cs train {i}') for i in range(14)],
        'commonsense_qa_validation': [qa(f'cs validation {i}') for i in range(8)],
        'arc_easy_train': [qa(f'arc train {i}', '2', ['1', '2', '3']) for i in range(14)],
        'arc_easy_validation': [qa(f'arc validation {i}') for i in range(8)],
        'arc_easy_test': [qa(f'arc test {i}') for i in range(8)],
    }
    # Repeat an entire official evaluation pool in another dataset's training pool.
    sources['skywork_easy_math'] += [sky(f'g test {i}') for i in range(8)]
    sources['commonsense_qa_train'].append(sources['commonsense_qa_train'][0])
    (root/'data').mkdir()
    for name, rows in sources.items():
        (root/'data'/f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return dict(seed=31, datasets=specs)


def test_choice_adapter_preserves_numeric_arc_labels():
    example = normalize_row('arc_easy', 'train', 2, qa('Which?', '2', ['1', '2', '3']), {})
    assert example['gold'] == '2'
    assert example['choice_labels'] == ['1', '2', '3']
    assert '1. option 1' in example['question']
    assert 'label (1, 2, 3)' in format_instruction(example)
    assert 'numeric answer' not in format_instruction(example)
    assert example['question_hash'] == question_hash('Which?')
    with pytest.raises(ValueError, match='Invalid choices'):
        normalize_row('arc_easy', 'train', 2, qa('Which?', 'Z'), {})
    assert normalize_row('commonsense_qa', 'test', 2, qa('Which?', ''), {}) is None


@pytest.mark.parametrize('text,expected', [
    (r'B</think>\boxed{A}', 'A'),
    (r'B</think>\boxed{b}', 'B'),
    (r'B</think>Final answer: (C).', 'C'),
    (r'B</think>\boxed{\text{B}}', 'B'),
    (r'B</think>\boxed{A}\boxed{C}', 'C'),
    (r'\boxed{B}', None),
    (r'B</think>The answer may be B.', None),
    (r'B</think>\boxed{B or C}', None),
    (r'B</think>Final answer: B or C', None),
    (r'B</think>\boxed{D}', None),
])
def test_choice_scoring_requires_explicit_unambiguous_final_label(text, expected):
    assert extract_answer(text, 'choice', ['A', 'B', 'C']) == expected


class Tokenizer:
    eos_token_id = 9
    def __init__(self, text): self.text = text
    def convert_tokens_to_ids(self, token): return {'</think>': 3, '<|im_end|>': 9}[token]
    def decode(self, ids, **kwargs): return self.text


def test_choice_answers_require_termination_and_correct_label():
    tokenizer = Tokenizer(r'reason</think>\boxed{2}')
    kwargs = dict(answer_type='choice', choice_labels=['1', '2', '3'])
    assert score_tokens([1, 3, 4, 9], tokenizer, '2', 3, **kwargs)['correct']
    assert not score_tokens([1, 3, 4], tokenizer, '2', 3, **kwargs)['correct']
    assert not score_tokens([1, 3, 4, 9], tokenizer, '1', 3, **kwargs)['correct']


def test_skywork_band_boundaries_and_missing_difficulty():
    spec = dict(difficulty_min=1, difficulty_max=4)
    for difficulty in [1, 4]:
        row = normalize_row('skywork_easy', 'math', 0, sky('q', difficulty, ['3/2', '1.5']), spec)
        assert row['accepted_answers'] == ['3/2', '1.5']
        assert row['source_difficulty'] == difficulty
        assert 'numeric answer' not in format_instruction(row)
    for difficulty in [0, 5, 16, None]:
        assert normalize_row('skywork_easy', 'math', 0, sky('q', difficulty), spec) is None
    assert normalize_row('skywork_easy', 'math', 0, sky('q'), dict(difficulty_model='absent')) is None


def test_math_nested_boxes_rationals_alternatives_and_conservative_equivalence():
    assert normalize_math(r'\\frac{3}{2}') == '3/2'
    assert normalize_math(r'\dfrac{3}{2}') == '3/2'
    assert normalize_math('1,2') == '1,2'
    assert normalize_math('1,234') == '1234'
    assert extract_answer(r'</think>\boxed{\frac{\sqrt{3}}{2}}', 'math') == r'\frac{\sqrt{3}}{2}'
    kwargs = dict(answer_type='math', accepted_answers=['1', r'\\frac{3}{2}'])
    assert score_tokens([1, 3, 4, 9], Tokenizer(r'</think>\boxed{1.5}'), '1', 3, **kwargs)['correct']
    assert not score_tokens([1, 3, 4, 9], Tokenizer(r'</think>\boxed{x+x}'), '2x', 3, answer_type='math')['correct']


def test_split_counts_determinism_and_global_holdout_exclusion(tmp_path):
    config = corpus(tmp_path)
    summary = {}
    splits = build_splits(tmp_path, config, summary)
    assert splits == build_splits(tmp_path, config)
    assert summary['commonsense_qa']['train'] == 14  # duplicate removed
    for split, count in [('train', 3), ('validation', 2), ('test', 2)]:
        assert len(splits[split]) == 4*count
        for spec in config['datasets']:
            assert sum(x['dataset'] == spec['name'] for x in splits[split]) == count
    hashes = [x['question_hash'] for rows in splits.values() for x in rows]
    assert len(hashes) == len(set(hashes))
    assert not any(x['question_hash'] == question_hash(f'g test {i}') for x in splits['train'] for i in range(8))
    assert all(x['source_split'] == 'validation' for x in splits['test'] if x['dataset'] == 'commonsense_qa')
    assert all(x['source_split'] == 'train' for x in splits['validation'] if x['dataset'] == 'commonsense_qa')
    assert all(x['source_split'] == 'validation' for x in splits['validation'] if x['dataset'] == 'arc_easy')
    assert splits != build_splits(tmp_path, {**config, 'seed': 32})


def test_missing_data_and_insufficient_pool_fail_loudly(tmp_path):
    config = corpus(tmp_path)
    config['datasets'][1]['train_n'] = 500
    with pytest.raises(ValueError, match='skywork_easy/train: requested 500'):
        build_splits(tmp_path, config)
    (tmp_path/'data/arc_easy_test.jsonl').unlink()
    with pytest.raises(FileNotFoundError, match='prepare.py'):
        build_splits(tmp_path, config)


def test_dataset_file_hash_verified_before_sampling(tmp_path):
    config = corpus(tmp_path)
    write_json(tmp_path/'manifest.json', dict(datasets=dict(train=dict(sha256='bad'))))
    with pytest.raises(RuntimeError, match='Dataset hash mismatch: gsm8k/train'):
        load_pools(tmp_path, config['datasets'][0])


@pytest.mark.parametrize('specs', [
    [dict(name='unknown', train_n=1, val_n=1, test_n=1)],
    [dict(name='gsm8k', train_n=-1, val_n=1, test_n=1)],
    [dict(name='skywork_easy', train_n=1, val_n=1, test_n=1, difficulty_min=5, difficulty_max=1)],
    [dict(name='gsm8k', train_n=1, val_n=1, test_n=1)]*2,
])
def test_invalid_dataset_configuration(specs):
    with pytest.raises(ValueError): dataset_specs(dict(datasets=specs))


def test_training_keeps_task_metadata_and_masks_the_same_qa_prompt(tmp_path, monkeypatch):
    import experiment as e
    config = corpus(tmp_path)
    example = next(x for x in build_splits(tmp_path, config)['train'] if x['answer_type'] == 'choice')
    monkeypatch.setattr(e, 'RUN', tmp_path/'run')
    monkeypatch.setattr(e, 'log', lambda message: None)
    monkeypatch.setattr(e, 'C', dict(train_max_length=10000))
    class ChatTokenizer:
        def apply_chat_template(self, messages, **kwargs): return messages[0]['content']+'\n<think>'
        def encode(self, text, **kwargs): return [ord(x) for x in text]
    monkeypatch.setattr(e, 'TOKENIZER', ChatTokenizer())
    completion = r'Consider the relevant fact.</think>\boxed{'+example['gold']+'}'
    candidate = {**example, 'correct': True, 'thinking_closed': True, 'total_tokens': 20, 'text': completion}
    targets = e.select_targets([example], [{'total_tokens': 100}], [{example['id']: candidate}])
    assert targets[0]['choice_labels'] == example['choice_labels']
    assert targets[0]['answer_type'] == 'choice'
    assert targets[0]['dataset'] == example['dataset']
    prefix = e.prompt(example)
    assert 'numeric answer' not in prefix
    assert 'Teacher coaching' not in prefix
    encoded = e.training_examples(targets)[0]
    assert encoded['labels'][:len(prefix)] == [-100]*len(prefix)
    assert encoded['input_ids'][:len(prefix)] == [ord(c) for c in prefix]
    assert encoded['labels'][len(prefix):] == [ord(c) for c in completion+'<|im_end|>']


def test_reports_include_domain_pairing_and_partial_completion(tmp_path, monkeypatch):
    import experiment as e
    from common import append_jsonl
    monkeypatch.setattr(e, 'RUN', tmp_path)
    monkeypatch.setattr(e, 'C', dict(datasets=[dict(name='gsm8k'), dict(name='commonsense_qa')]))
    def row(id, dataset, correct, thinking):
        return dict(id=id, dataset=dataset, task='math' if dataset=='gsm8k' else 'commonsense_qa', difficulty='unrated',
                    correct=correct, thinking_tokens=thinking, total_tokens=thinking+2, thinking_closed=True,
                    finished=True, truncated=False, parsed_answer='B', seconds=1)
    base = [row('g1', 'gsm8k', True, 10), row('c1', 'commonsense_qa', True, 20), row('c2', 'commonsense_qa', False, 30)]
    current = [row('g1', 'gsm8k', False, 5), row('c1', 'commonsense_qa', True, 5)]
    for arm, rows in [('baseline', base), ('mlp', current)]:
        for r in rows: append_jsonl(tmp_path/f'test_{arm}.jsonl', r)
    write_json(tmp_path/'splits.json', dict(test=base))
    metrics = e.report()['test']
    assert metrics['baseline']['macro_dataset_accuracy'] == .75
    assert metrics['mlp']['datasets']['gsm8k']['paired_comparison']['accuracy_lost'] == 1
    assert metrics['mlp']['datasets']['commonsense_qa']['paired_comparison']['n'] == 1
    assert metrics['mlp']['datasets']['commonsense_qa']['expected_n'] == 2
    assert not metrics['mlp']['complete']
    assert metrics['mlp']['tasks']['commonsense_qa']['accuracy'] == 1
    report = (tmp_path/'RESULTS.md').read_text()
    assert 'Per dataset' in report and 'Mixed-domain' in report
    assert 'not transfer to an unseen training domain' in report


def test_pinned_parquet_preparation_and_hash_guard(tmp_path, monkeypatch):
    import prepare
    import pyarrow as pa
    import pyarrow.parquet as pq
    monkeypatch.setattr(prepare, 'ROOT', tmp_path)
    source = tmp_path/'source.parquet'
    pq.write_table(pa.Table.from_pylist([qa('a'), qa('b')]), source)
    class API:
        def dataset_info(self, repo): return type('Info', (), dict(sha='pinned-sha'))()
        def list_repo_files(self, repo, **kwargs):
            assert kwargs['revision'] == 'pinned-sha'
            return ['data/train-00000-of-00001.parquet', 'data/validation-00000-of-00001.parquet']
    calls = []
    def download(repo, filename, **kwargs):
        assert kwargs['revision'] == 'pinned-sha'
        calls.append(filename)
        return source
    monkeypatch.setattr(prepare, 'hf_hub_download', download)
    manifest = dict(models={}, datasets={})
    prepare.prepare_hf_dataset('commonsense_qa', manifest, API())
    assert len(calls) == 2
    assert manifest['datasets']['commonsense_qa']['splits']['train']['rows'] == 2
    prepare.prepare_hf_dataset('commonsense_qa', manifest, API())
    assert len(calls) == 2
    with (tmp_path/'data/commonsense_qa_train.jsonl').open('a') as output: output.write('{}\n')
    with pytest.raises(RuntimeError, match='Dataset hash mismatch'):
        prepare.prepare_hf_dataset('commonsense_qa', manifest, API())
