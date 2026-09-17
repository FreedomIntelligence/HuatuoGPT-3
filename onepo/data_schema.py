"""Public task schema shared by training and evaluation."""
from __future__ import annotations

import json
from pathlib import Path

MC_INSTRUCTION = (
    'Answer the multiple-choice question and place the final selection within the <answer> tag. '
    'Example: <answer>A</answer>.'
)


def task_type(row):
    aliases = {'mc': 'multiple_choice', 'multiple_choice': 'multiple_choice',
               'openend': 'open_ended', 'open_ended': 'open_ended'}
    value = row.get('type')
    if value is None:
        return 'multiple_choice' if row.get('options') else 'open_ended'
    if value not in aliases:
        raise ValueError(f'Unknown task type: {value!r}')
    return aliases[value]


def choice_options(row):
    # HF Arrow schemas can fill absent option labels with null.
    options = row.get('options') or {}
    if not isinstance(options, dict):
        raise ValueError('options must be a label-to-text object')
    return {key: value for key, value in options.items() if value is not None}


def build_prompt(row):
    """Preserve existing prompts; construct MC chat input when prompt is null."""
    prompt = row.get('prompt')
    if prompt:
        if isinstance(prompt, str):
            return [{'role': 'user', 'content': prompt}]
        if not isinstance(prompt, list) or not all(
            isinstance(m, dict) and isinstance(m.get('role'), str) and isinstance(m.get('content'), str)
            for m in prompt
        ):
            raise ValueError('prompt must be a list of role/content messages')
        return prompt
    if task_type(row) == 'multiple_choice':
        question, options = row.get('question'), choice_options(row)
        if not isinstance(question, str) or not question.strip() or not options:
            raise ValueError('Multiple-choice records require question and options')
        if not all(isinstance(value, str) and value.strip() for value in options.values()):
            raise ValueError('Options must contain nonempty text')
        # Sort labels because Arrow may normalize the order of struct fields.
        choices = '\n'.join(f'{key}. {options[key]}' for key in sorted(options))
        return [{'role': 'user', 'content': f'{MC_INSTRUCTION}\n{question}\n{choices}'}]
    raise ValueError('Open-ended records require a nonempty prompt')


def load_records(path):
    path = Path(path)
    with path.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()] if path.suffix == '.jsonl' else json.load(stream)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f'Expected an array of records in {path}')
    return rows
