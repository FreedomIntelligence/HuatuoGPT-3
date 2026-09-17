"""OnePO mixed medical dataset adapter.

normalizes the paper's JSON records into
the current verl RL dataset contract and carries cached teacher responses.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

from onepo.data_schema import build_prompt, choice_options, task_type


class OnePODataset(RLHFDataset):
    """Load JSON/JSONL/Parquet samples for mixed rubric and MC training."""

    @staticmethod
    def _normalize_record(row: dict[str, Any], index: int) -> dict[str, Any]:
        prompt = build_prompt(row)
        is_mc = task_type(row) == "multiple_choice"
        if is_mc:
            ground_truth = {
                "type": "mc",
                "answer_idx": str(row.get("answer_idx", "")).upper(),
                "options": choice_options(row),
                "question": row.get("question", ""),
            }
        else:
            ground_truth = {"type": "openend", "rubrics": row.get("rubrics") or [], "prompt": prompt}
        extra_info = dict(row.get("extra_info") or {})
        extra_info.setdefault("index", index)
        return {
            "prompt_json": json.dumps(prompt, ensure_ascii=False),
            "ground_truth_json": json.dumps(ground_truth, ensure_ascii=False),
            "teacher_response_json": json.dumps(
                row.get("model_response", row.get("teacher_response")), ensure_ascii=False
            ),
            "data_source": str(row.get("data_source", row.get("source", "onepo_medical"))),
            "extra_info_json": json.dumps(extra_info, ensure_ascii=False),
            "index": int(extra_info["index"]) if str(extra_info["index"]).isdigit() else index,
        }

    def _read_files_and_tokenize(self) -> None:
        """Stream heterogeneous source records into a stable Arrow schema."""
        records = []
        for data_file in self.data_files:
            path = Path(data_file)
            if path.suffix == ".json":
                # ``json.load`` is intentional here: the medical source files
                # contain JSON-compatible NaN/Infinity sentinels that strict
                # streaming parsers reject.  ``parse_constant`` converts them
                # to null while preserving the rest of the record unchanged.
                with path.open(encoding="utf-8") as stream:
                    source = json.load(stream, parse_constant=lambda _value: None)
                if not isinstance(source, list):
                    raise ValueError(f"expected a JSON array in {path}")
                records.extend(self._normalize_record(row, len(records)) for row in source)
            elif path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            records.append(self._normalize_record(json.loads(line), len(records)))
            elif path.suffix == ".parquet":
                source = datasets.load_dataset("parquet", data_files=str(path))["train"]
                records.extend(self._normalize_record(row, len(records)) for row in source)
            else:
                raise ValueError(f"unsupported data file: {path}")

        if self.max_samples > 0 and self.max_samples < len(records):
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(len(records), size=self.max_samples, replace=False) if self.shuffle else range(self.max_samples)
            records = [records[int(i)] for i in indices]
        self.dataframe = datasets.Dataset.from_list(records)

        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            max_length = self.max_prompt_length

            def prompt_fits(row: dict[str, Any]) -> bool:
                encoded = tokenizer.apply_chat_template(
                    json.loads(row["prompt_json"]),
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                )
                return len(encoded["input_ids"]) <= max_length

            self.dataframe = self.dataframe.filter(
                prompt_fits,
                num_proc=self.num_workers,
                desc="Filtering overlong OnePO prompts",
            )
        print(f"OnePO dataset len: {len(self.dataframe)}")

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = dict(self.dataframe[item])
        prompt = json.loads(row["prompt_json"])
        ground_truth = json.loads(row["ground_truth_json"])
        teacher = json.loads(row["teacher_response_json"])
        if isinstance(teacher, list):
            teacher = random.choice(teacher) if teacher else None
        extra_info = json.loads(row["extra_info_json"])
        is_mc = ground_truth["type"] == "mc"
        return {
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "raw_prompt": prompt,
            "prompt": prompt,
            "data_source": row["data_source"],
            "reward_model": {
                "ground_truth": ground_truth,
                "style": "mc" if is_mc else "rubric",
            },
            "teacher_response": teacher,
            "index": row["index"],
            "extra_info": extra_info,
            "tools_kwargs": extra_info.get("tools_kwargs", {}),
            "interaction_kwargs": extra_info.get("interaction_kwargs", {}),
        }
