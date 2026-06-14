from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        prefill_chunk_size: int,
        mask_id: int,
        max_running_requests: int,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.prefill_chunk_size = prefill_chunk_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        from sglang.srt.configs.model_config import ModelConfig

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
            "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
        }

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        max_running_requests = (
            1
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f)
            if algorithm_config is None:
                algorithm_config = {}

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        prefill_chunk_size = algorithm_config.get("prefill_chunk_size", block_size)
        if server_args.dllm_prefill_chunk_size is not None:
            prefill_chunk_size = server_args.dllm_prefill_chunk_size
        if prefill_chunk_size < block_size:
            raise ValueError(
                f"dllm_prefill_chunk_size ({prefill_chunk_size}) must be greater "
                f"than or equal to dLLM block_size ({block_size})."
            )
        if prefill_chunk_size % block_size != 0:
            raise ValueError(
                f"dllm_prefill_chunk_size ({prefill_chunk_size}) must be a "
                f"multiple of dLLM block_size ({block_size})."
            )

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            prefill_chunk_size=prefill_chunk_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
        )
