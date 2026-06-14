import ast
import importlib.util
import sys
import tempfile
import types
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module_from_path(
    "ci_register", CI_REGISTER_PATH
).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


model_config_module = types.ModuleType("sglang.srt.configs.model_config")
model_config_module.ModelConfig = type("ModelConfig", (), {})
sys.modules["sglang.srt.configs.model_config"] = model_config_module

dllm_config_module = _load_module(
    "sglang.srt.dllm.config", "python/sglang/srt/dllm/config.py"
)
req_module = _load_module(
    "sglang.srt.dllm.mixin.req", "python/sglang/srt/dllm/mixin/req.py"
)

DllmConfig = dllm_config_module.DllmConfig
ReqDllmMixin = req_module.ReqDllmMixin


def _server_args(**overrides):
    values = dict(
        dllm_algorithm="block_diffusion",
        model_path="dummy",
        revision=None,
        max_running_requests=None,
        dllm_algorithm_config=None,
        dllm_prefill_chunk_size=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _model_config(arch="LLaDA2MoeModelLM"):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))


class FakeMixinReq(ReqDllmMixin):
    pass


class TestDllmPrefillChunkConfig(unittest.TestCase):
    @patch.object(model_config_module.ModelConfig, "from_server_args", create=True)
    def test_default_prefill_chunk_size_is_block_size(self, mock_model_config):
        mock_model_config.return_value = _model_config()

        config = DllmConfig.from_server_args(_server_args())

        self.assertEqual(config.block_size, 32)
        self.assertEqual(config.prefill_chunk_size, 32)

    @patch.object(model_config_module.ModelConfig, "from_server_args", create=True)
    def test_yaml_prefill_chunk_size_and_cli_override(self, mock_model_config):
        mock_model_config.return_value = _model_config()
        with tempfile.NamedTemporaryFile("w") as f:
            f.write("prefill_chunk_size: 64\n")
            f.flush()

            config = DllmConfig.from_server_args(
                _server_args(dllm_algorithm_config=f.name)
            )
            override_config = DllmConfig.from_server_args(
                _server_args(dllm_algorithm_config=f.name, dllm_prefill_chunk_size=96)
            )

        self.assertEqual(config.prefill_chunk_size, 64)
        self.assertEqual(override_config.prefill_chunk_size, 96)

    @patch.object(model_config_module.ModelConfig, "from_server_args", create=True)
    def test_prefill_chunk_size_must_be_block_multiple(self, mock_model_config):
        mock_model_config.return_value = _model_config()

        with self.assertRaisesRegex(ValueError, "multiple"):
            DllmConfig.from_server_args(_server_args(dllm_prefill_chunk_size=48))

        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            DllmConfig.from_server_args(_server_args(dllm_prefill_chunk_size=16))

    @patch.object(model_config_module.ModelConfig, "from_server_args", create=True)
    def test_empty_yaml_config_is_allowed(self, mock_model_config):
        mock_model_config.return_value = _model_config()
        with tempfile.NamedTemporaryFile("w") as f:
            f.flush()

            config = DllmConfig.from_server_args(
                _server_args(dllm_algorithm_config=f.name)
            )

        self.assertEqual(config.block_size, 32)
        self.assertEqual(config.prefill_chunk_size, 32)


class TestDllmRequestOffset(unittest.TestCase):
    def test_dllm_block_offset_advances_by_previous_extend_len(self):
        req = FakeMixinReq()
        req.fill_len = 64
        req.dllm_block_offset = 0
        req.extend_input_len = 64
        req.dllm_config = SimpleNamespace(block_size=32, mask_id=999)
        req.origin_input_ids = array("q", range(96))
        req.output_ids = array("q")

        req._init_fill_ids_for_dllm()

        self.assertEqual(req.dllm_block_offset, 64)

    def test_phase_uses_next_semantic_block_only(self):
        req = FakeMixinReq()
        req.prefix_indices = array("q", range(32))
        req.dllm_config = SimpleNamespace(block_size=32, mask_id=999)
        req.full_untruncated_fill_ids = array("q", range(64))
        req.full_untruncated_fill_ids.extend([999])

        req.determine_dllm_phase()

        self.assertTrue(req.is_dllm_prefill())

    def test_phase_switches_to_decode_when_next_block_has_mask(self):
        req = FakeMixinReq()
        req.prefix_indices = array("q", range(32))
        req.dllm_config = SimpleNamespace(block_size=32, mask_id=999)
        req.full_untruncated_fill_ids = array("q", range(32))
        req.full_untruncated_fill_ids.extend([999])
        req.full_untruncated_fill_ids.extend(range(33, 64))

        req.determine_dllm_phase()

        self.assertFalse(req.is_dllm_prefill())


class TestDllmSchedulerPrefillChunk(unittest.TestCase):
    def test_dllm_incoming_reqs_do_not_use_prefill_truncation_align(self):
        scheduler_path = REPO_ROOT / "python/sglang/srt/dllm/mixin/scheduler.py"
        module = ast.parse(scheduler_path.read_text())

        target_call = None
        for node in ast.walk(module):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "process_dllm_incoming_reqs":
                continue
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "add_one_req"
                ):
                    target_call = child
                    break

        self.assertIsNotNone(target_call)
        truncation_arg = next(
            kw for kw in target_call.keywords if kw.arg == "truncation_align_size"
        )
        self.assertIsInstance(truncation_arg.value, ast.Constant)
        self.assertIsNone(truncation_arg.value.value)


if __name__ == "__main__":
    unittest.main()
