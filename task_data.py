"""Dataset adapters and deterministic, cross-dataset-disjoint experiment splits."""
import hashlib
import json
import random
from pathlib import Path

DATASETS = {
    'gsm8k': {'repo': 'openai/gsm8k', 'config': 'main', 'splits': ['train', 'test']},
    'skywork_easy': {'repo': 'Skywork/Skywork-OR1-RL-Data', 'config': 'default', 'splits': ['math']},
    'commonsense_qa': {'repo': 'tau/commonsense_qa', 'config': 'default', 'splits': ['train', 'validation']},
    'arc_easy': {'repo': 'allenai/ai2_arc', 'config': 'ARC-Easy', 'splits': ['train', 'validation', 'test']},
}


def question_hash(question):
    return hashlib.sha256(' '.join(question.casefold().split()).encode()).hexdigest()


def dataset_specs(config):
    specs = config.get('datasets', [])
    names = [s['name'] for s in specs]
    if len(names) != len(set(names)):
        raise ValueError('Dataset names must be unique')
    for spec in specs:
        if spec['name'] not in DATASETS:
            raise ValueError(f"Unknown dataset: {spec['name']}")
        for field in ['train_n', 'val_n', 'test_n']:
            if not isinstance(spec.get(field), int) or isinstance(spec[field], bool) or spec[field] < 0:
                raise ValueError(f'{field} must be a nonnegative integer for {spec["name"]}')
        if spec['name'] == 'skywork_easy':
            lo, hi = spec.get('difficulty_min', 1), spec.get('difficulty_max', 4)
            if not isinstance(lo, int) or not isinstance(hi, int) or not 0 <= lo <= hi <= 16:
                raise ValueError('Skywork difficulty must satisfy 0 <= min <= max <= 16')
    return specs


def normalize_row(name, split, index, row, spec):
    """Keep source IDs/labels; render options without exposing the answer key."""
    base = dict(id=f'{name}-{split}-{index}', dataset=name, source_split=split,
                source_id=str(row.get('id', index)))
    if name == 'gsm8k':
        stem = row['question']
        if '####' not in row['answer']:
            raise ValueError('Missing GSM8K gold delimiter')
        steps = row['answer'].count('<<')
        base.update(question=stem, gold=row['answer'].rsplit('####', 1)[1].strip(),
                    task='math', answer_type='numeric',
                    difficulty='easy' if steps <= 2 else 'medium' if steps <= 4 else 'hard')
    elif name in ('commonsense_qa', 'arc_easy'):
        stem = row['question']
        labels, texts = list(row['choices']['label']), list(row['choices']['text'])
        gold = row.get('answerKey', '')
        if not gold:  # CommonsenseQA's public test has no labels; never score it.
            return None
        if not labels or len(labels) != len(texts) or len(labels) != len(set(labels)) or gold not in labels:
            raise ValueError(f'Invalid choices or answer key: {name}/{split}/{index}')
        base.update(question=stem+'\n\nChoices:\n'+'\n'.join(f'{label}. {text}' for label, text in zip(labels, texts)),
                    choices=dict(label=labels, text=texts), choice_labels=labels, gold=gold,
                    task='commonsense_qa' if name == 'commonsense_qa' else 'science_qa',
                    answer_type='choice', difficulty='unrated')
    elif name == 'skywork_easy':
        model = spec.get('difficulty_model', 'DeepSeek-R1-Distill-Qwen-1.5B')
        difficulty = row.get('extra_info', {}).get('model_difficulty', {}).get(model)
        if difficulty is None or not spec.get('difficulty_min', 1) <= difficulty <= spec.get('difficulty_max', 4):
            return None
        messages = row['prompt']
        if len(messages) != 1 or messages[0].get('role') != 'user':
            raise ValueError('Expected a single user prompt in Skywork math')
        stem = messages[0]['content']
        answers = row['reward_model']['ground_truth']
        if isinstance(answers, str):
            answers = json.loads(answers)
        if not isinstance(answers, list) or not answers or not all(isinstance(a, str) and a.strip() for a in answers):
            raise ValueError('Invalid Skywork reference answers')
        answers = [a.strip() for a in answers]
        base.update(question=stem, gold=answers[0], accepted_answers=answers, task='math', answer_type='math',
                    difficulty='easy', source_difficulty=difficulty, difficulty_model=model)
    else:
        raise ValueError(f'Unknown dataset: {name}')
    base['question_hash'] = question_hash(stem)
    return base


def load_pools(root, spec):
    name = spec['name']
    pools = {}
    manifest_path = Path(root)/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    for split in DATASETS[name]['splits']:
        path = Path(root)/'data'/f'{name}_{split}.jsonl'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}; run prepare.py with this configuration first')
        if manifest is not None:
            entries = manifest.get('datasets', {})
            saved = entries.get(split) if name == 'gsm8k' else entries.get(name, {}).get('splits', {}).get(split)
            if not saved:
                raise RuntimeError(f'Unverified dataset: {name}/{split}; run prepare.py first')
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != saved['sha256']:
                raise RuntimeError(f'Dataset hash mismatch: {name}/{split}')
        seen = set()
        rows = []
        with path.open() as stream:
            for i, line in enumerate(stream):
                if not line.strip():
                    continue
                example = normalize_row(name, split, i, json.loads(line), spec)
                if example is None or example['question_hash'] in seen:
                    continue
                seen.add(example['question_hash'])
                rows.append(example)
        pools[split] = rows
    return pools


def build_splits(root, config, summary=None):
    specs = dataset_specs(config)
    if not specs:
        raise ValueError('Mixed runs require a nonempty datasets list')
    pools = {s['name']: load_pools(root, s) for s in specs}
    if summary is not None:
        summary.update({name: {split: len(rows) for split, rows in sources.items()} for name, sources in pools.items()})
    rng = random.Random(config['seed'])
    for sources in pools.values():
        for rows in sources.values():
            rng.shuffle(rows)
    result = {key: [] for key in ['train', 'validation', 'test']}
    used = set()
    # Reserve ALL official evaluation questions against training, across datasets.
    official_eval = {x['question_hash'] for name, sources in pools.items() for split, rows in sources.items()
                     if split in ('validation', 'test') for x in rows}

    def take(name, source, target, count, blocked=()):
        eligible = [x for x in pools[name][source] if x['question_hash'] not in used and x['question_hash'] not in blocked]
        if len(eligible) < count:
            raise ValueError(f'{name}/{target}: requested {count}, only {len(eligible)} eligible unique examples')
        chosen = eligible[:count]
        used.update(x['question_hash'] for x in chosen)
        result[target].extend(chosen)

    # Official labeled evaluation sets first. CommonsenseQA validation serves as test.
    for spec in specs:
        name = spec['name']
        if name == 'skywork_easy':
            continue
        take(name, 'validation' if name == 'commonsense_qa' else 'test', 'test', spec['test_n'])
        if name == 'arc_easy':
            take(name, 'validation', 'validation', spec['val_n'])
    for spec in specs:
        if spec['name'] == 'skywork_easy':
            take('skywork_easy', 'math', 'test', spec['test_n'], official_eval)
    # Datasets without official validation use a disjoint training-pool holdout.
    for spec in specs:
        name = spec['name']
        if name != 'arc_easy':
            take(name, 'math' if name == 'skywork_easy' else 'train', 'validation', spec['val_n'], official_eval)
    for spec in specs:
        name = spec['name']
        take(name, 'math' if name == 'skywork_easy' else 'train', 'train', spec['train_n'], official_eval)
    for rows in result.values():
        rng.shuffle(rows)
    hashes = [x['question_hash'] for rows in result.values() for x in rows]
    assert len(hashes) == len(set(hashes)), 'Cross-dataset split leakage'
    return result
