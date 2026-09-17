"""OnePO metadata adapter for verl's unmodified DAPO reward manager."""

import numpy as np
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager


class OnePORewardManager(DAPORewardManager):
    """Derive thinking-prefix metadata from the actual prompt before scoring."""

    async def run_single(self, data: DataProto) -> dict:
        # Match DAPO's last-sequence convention; do not mutate shared input metadata.
        sample = data[-1:]
        item = sample[0]
        extra_info = dict(item.non_tensor_batch.get("extra_info") or {})
        extra_info["prompt_ends_with_think"] = False
        if "prompts" in item.batch:
            prompt_ids = item.batch["prompts"]
            prompt_mask = item.batch["attention_mask"][: len(prompt_ids)].bool()
            extra_info["prompt_ends_with_think"] = self.tokenizer.decode(
                prompt_ids[prompt_mask], skip_special_tokens=False
            ).rstrip().endswith("<think>")
        metadata = dict(sample.non_tensor_batch)
        metadata["extra_info"] = np.array([extra_info], dtype=object)
        sample = DataProto(batch=sample.batch, non_tensor_batch=metadata, meta_info=sample.meta_info)
        return await super().run_single(sample)
