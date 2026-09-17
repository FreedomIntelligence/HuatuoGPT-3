"""OnePO's single-turn rollout adapter for the pinned, unmodified verl."""

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class OnePOSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Select the processor-backed Qwen builder for unified Qwen3.5 models."""

    def __init__(
        self, trainer_config, server_manager, tokenizer, processor, dataset_cls,
        data_config, hf_model_type=None, **kwargs,
    ):
        builder_model_type = hf_model_type
        model_type = hf_model_type.strip().lower() if isinstance(hf_model_type, str) else None
        if model_type in {"qwen3_5", "qwen3_5_moe"} and getattr(processor, "image_processor", None) is not None:
            # This argument only selects the token builder in AgentLoopBase.
            # Preserve the actual model config, tokenizer, processor, and weights.
            builder_model_type = "qwen3_vl"
        super().__init__(
            trainer_config=trainer_config, server_manager=server_manager,
            tokenizer=tokenizer, processor=processor, dataset_cls=dataset_cls,
            data_config=data_config, hf_model_type=builder_model_type, **kwargs,
        )
