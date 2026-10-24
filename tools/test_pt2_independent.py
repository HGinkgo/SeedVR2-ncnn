# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 SeedVR2-ncnn contributors
"""Reproduce the independent PT2 frontend against TorchScript and ncnn."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.convert_pt2 import convert, validate_native_layers
from tools.export_pt2_seedvr2 import (SeedVR2MLPProbe, SeedVR2MLPResidualProbe,
                                      SeedVR2AttentionCoreProbe,
                                      SeedVR2AdaEmbeddingProbe,
                                      load_checkpoint_video_ada, load_checkpoint_video_mlp,
                                      load_checkpoint_video_ada_mlp,
                                      load_checkpoint_video_mlp_residual,
                                      load_checkpoint_video_attention_core,
                                      load_checkpoint_video_ada_attention,
                                      load_checkpoint_video_ada_attention_gate)
from tools.export_dit_block import rewrite_ncnn_param
from tools.reference.dit_block import DIT_BLOCK_CONTRACT, load_fixed_dit_block
from tools.reference.dit_block import (FixedAdaBranch, FixedRMSNorm, FixedSwiGLU,
                                      build_mmrope_frequencies, _apply_fixed_rotary)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _outputs(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _load_generated(path):
    spec = importlib.util.spec_from_file_location("pnnx_generated", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated PNNX Python: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model().eval()


def _assert_close(label, actual, expected, tolerance=1e-5):
    actual_values = _outputs(actual)
    expected_values = _outputs(expected)
    if len(actual_values) != len(expected_values):
        raise AssertionError(f"{label}: output count differs")
    if any(a.shape != b.shape for a, b in zip(actual_values, expected_values)):
        raise AssertionError(f"{label}: output shape differs")
    if any(not bool(torch.isfinite(a).all()) for a in actual_values):
        raise AssertionError(f"{label}: output contains non-finite values")
    errors = [float((a.detach() - b.detach()).abs().max()) for a, b in zip(actual_values, expected_values)]
    if any(error > tolerance for error in errors):
        raise AssertionError(f"{label}: max absolute errors {errors} exceed {tolerance}")
    print(f"{label}: max_abs={max(errors, default=0.0):.8g}")
    return max(errors, default=0.0)


def _conversion_failure_contracts(work, pt2, converter=convert, pnnx=Path("/unused/pnnx")):
    """A failed retry must never reuse old native output as successful evidence."""
    destination = work / "failed-conversion"
    destination.mkdir(exist_ok=True)
    stale = destination / f"{pt2.stem}.ncnn.param"
    stale.write_text("previous model")
    with patch("tools.convert_pt2.subprocess.run") as launch:
        try:
            converter(pt2, pnnx, destination)
        except ValueError as exc:
            assert "already exist" in str(exc)
        else:
            raise AssertionError("conversion overwrote existing artifacts without explicit intent")
        launch.assert_not_called()
    assert stale.read_text() == "previous model"
    unknown = work / "unknown-layer.param"
    unknown.write_text("7767517\n1 1\nunregistered.operator op 0 1 out0\n")
    try:
        validate_native_layers(unknown)
    except ValueError as exc:
        assert "unsupported/unlowered ncnn layer types" in str(exc)
    else:
        raise AssertionError("unconverted operator accepted as a native ncnn layer")
    for returncode, text, message in ((17, "operator rejected\n", "exit code 17"),
                                       (0, "converter wrote nothing\n", "without expected artifacts")):
        stale.write_text("previous model")
        with patch("tools.convert_pt2.subprocess.run", return_value=
                   subprocess.CompletedProcess([], returncode, text)):
            try:
                converter(pt2, pnnx, destination, overwrite=True)
            except RuntimeError as exc:
                assert message in str(exc)
            else:
                raise AssertionError("conversion accepted a failed or incomplete result")
        record = json.loads((destination / f"{pt2.stem}.conversion.json").read_text())
        assert record["status"] == "failed" and record["returncode"] == returncode
        assert record["artifacts"] == [] and not stale.exists()
        assert Path(record["log"]).read_text() == text


def _unsupported_programs(work, pnnx, converter=convert):
    """Ensure unsupported ExportedProgram calling conventions fail before conversion."""
    class Nested(torch.nn.Module):
        def forward(self, values):
            return values["x"] + 1

    class Scalar(torch.nn.Module):
        def forward(self, value, scale):
            return value * scale

    class ControlFlow(torch.nn.Module):
        def forward(self, value):
            return torch.cond(value.sum() > 0, lambda x: x + 1, lambda x: x - 1, (value,))

    class Mutation(torch.nn.Module):
        def forward(self, value):
            return value.add_(1)

    value = torch.ones(2, 4)
    for name, model, args, message in (
        ("nested", Nested(), ({"x": value},), "nested or keyword inputs"),
        ("scalar", Scalar(), (value, 2), "user input must be a tensor"),
        ("control", ControlFlow(), (value,), "control flow/higher-order"),
        ("mutation", Mutation(), (value.clone(),), "mutation/state output"),
    ):
        program = torch.export.export(model, args).run_decompositions()
        path = work / f"unsupported-{name}.pt2"
        torch.export.save(program, path)
        try:
            converter(path, pnnx, work / f"unsupported-{name}")
        except ValueError as exc:
            if message not in str(exc):
                raise
        else:
            raise AssertionError(f"unsupported {name} program was accepted")


def run_ada_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures):
    """Validate the checkpoint-backed Ada branch with three rank-2 inputs."""
    directory = work / 'checkpoint-ada'
    directory.mkdir(exist_ok=True)
    command = [sys.executable, str(repo / 'tools/export_pt2_seedvr2.py'),
               '--model', 'checkpoint-ada', '--format', 'both', '--dynamic',
               '--checkpoint', str(args.checkpoint_mlp.resolve()), '--output-dir', str(directory)]
    subprocess.run(command, check=True)
    stem = 'dit_block0_video_ada_attn'
    model_pt2 = directory / (stem + '.pt2')
    record = convert(model_pt2, pnnx, directory / 'converted', overwrite=True)
    contract = record['frontend_contract']
    reference = torch.load(directory / (stem + '_reference.pt'), weights_only=True)
    eager = load_checkpoint_video_ada(args.checkpoint_mlp)[0]
    expected_state = eager.state_dict()
    eager.load_state_dict(expected_state)
    exported = torch.export.load(model_pt2)
    assert set(exported.state_dict) == set(expected_state)
    for key, value in exported.state_dict.items():
        torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
        entry = next(v for v in contract['weights'].values() if v['target'] == key)
        assert entry['sha256'] == hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
    assert len(contract['inputs']) == 3 and len(contract['outputs']) == 1
    assert 'aten.add.Tensor' in contract['operators'] and 'aten.mul.Tensor' in contract['operators']
    generated = _load_generated(directory / 'converted' / (stem + '_pnnx.py'))

    ts_dir = directory / 'torchscript'
    ts_dir.mkdir(exist_ok=True)
    ts = ts_dir / (stem + '.pt')
    shutil.copyfile(directory / (stem + '.pt'), ts)
    subprocess.run([str(pnnx), str(ts), 'inputshape=[7,2560],[7,2560],[7,2560]', 'fp16=0'],
                   check=True, cwd=ts_dir)
    ts_generated = _load_generated(ts_dir / (stem + '_pnnx.py'))
    validate_native_layers(ts_dir / (stem + '.ncnn.param'))
    with torch.no_grad():
        for tokens in (7, 13):
            values = reference['inputs'] if tokens == 7 else tuple(
                torch.randn(tokens, 2560, generator=torch.Generator().manual_seed(20260823 + tokens + index))
                for index in range(3))
            expected = eager(*values)
            torch.testing.assert_close(exported.module()(*values), expected, rtol=0, atol=0)
            error = _assert_close(f'checkpoint-ada independent PNNX tokens={tokens}',
                                  generated(*values), expected, 2e-5)
            python_results.append({'fixture': 'checkpoint-ada', 'tokens': tokens, 'max_abs': error})
            if tokens == 7:
                _assert_close('checkpoint-ada TorchScript PNNX baseline', ts_generated(*values), expected, 2e-5)
            if runner:
                for mode in (('cpu', 'vulkan') if args.vulkan else ('cpu',)):
                    for frontend, graph in ([('pt2', directory / 'converted')] +
                                            ([('torchscript', ts_dir)] if tokens == 7 else [])):
                        input_paths = []
                        for index, value in enumerate(values):
                            input_path = directory / f'{tokens}.input{index}.f32'
                            value.numpy().tofile(input_path)
                            input_paths.append(input_path)
                        output_path = directory / f'{frontend}-{tokens}-{mode}.output.f32'
                        prefix = [str(repo / 'tools/with-nvidia-vulkan-runtime.sh')] if mode == 'vulkan' else []
                        native_command = prefix + [str(runner), mode,
                            str(graph / (stem + '.ncnn.param')), str(graph / (stem + '.ncnn.bin')), '--multi', '3']
                        for index, input_path in enumerate(input_paths):
                            native_command.extend([f'in{index}', str(input_path), '2560', str(tokens)])
                        native_command.extend(['out0', str(output_path)])
                        started = time.perf_counter()
                        subprocess.run(native_command, check=True)
                        elapsed = time.perf_counter() - started
                        actual = torch.from_numpy(np.fromfile(output_path, dtype=np.float32)).reshape(expected.shape)
                        error = _assert_close(f'checkpoint-ada {frontend} ncnn {mode} tokens={tokens}',
                                              actual, expected, 2e-4)
                        native.append({'fixture': 'checkpoint-ada', 'frontend': frontend, 'mode': mode,
                                       'tokens': tokens, 'max_abs': error,
                                       'process_wall_seconds': elapsed, 'command': native_command})
    fixtures.append(json.loads((directory / (stem + '.manifest.json')).read_text()))


def run_ada_embedding_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures):
    """Validate packed embedding materialization against the fixed Ada reference."""
    directory = work / 'checkpoint-ada-embedding'
    directory.mkdir(exist_ok=True)
    command = [sys.executable, str(repo / 'tools/export_pt2_seedvr2.py'),
               '--model', 'checkpoint-ada-embedding', '--format', 'both', '--dynamic',
               '--checkpoint', str(args.checkpoint_mlp.resolve()), '--output-dir', str(directory)]
    subprocess.run(command, check=True)
    stem = 'dit_block0_video_ada_attn_embedding'
    model_pt2 = directory / (stem + '.pt2')
    record = convert(model_pt2, pnnx, directory / 'converted', overwrite=True)
    contract = record['frontend_contract']
    assert all(operator in contract['operators'] for operator in
               ('aten.reshape.default', 'aten.slice.Tensor', 'aten.select.int',
                'aten.add.Tensor', 'aten.mul.Tensor'))
    assert len(contract['inputs']) == 2 and len(contract['outputs']) == 1
    reference = torch.load(directory / (stem + '_reference.pt'), weights_only=True)
    exported = torch.export.load(model_pt2)
    exported_model = exported.module()
    exported_state = exported.state_dict
    probe = SeedVR2AdaEmbeddingProbe(2560, shift=exported_state['attn_shift'],
                                      scale=exported_state['attn_scale']).eval()
    fixed = FixedAdaBranch(2560).eval()
    with torch.no_grad():
        fixed.attn_shift.copy_(exported_state['attn_shift'])
        fixed.attn_scale.copy_(exported_state['attn_scale'])
    generated = _load_generated(directory / 'converted' / (stem + '_pnnx.py'))

    ts_dir = directory / 'torchscript'
    ts_dir.mkdir(exist_ok=True)
    ts = ts_dir / (stem + '.pt')
    shutil.copyfile(directory / (stem + '.pt'), ts)
    subprocess.run([str(pnnx), str(ts), 'inputshape=[7,2560],[1,15360]', 'fp16=0'],
                   check=True, cwd=ts_dir)
    ts_generated = _load_generated(ts_dir / (stem + '_pnnx.py'))
    validate_native_layers(ts_dir / (stem + '.ncnn.param'))
    with torch.no_grad():
        for tokens in (7, 13):
            values = reference['inputs'] if tokens == 7 else (
                torch.randn(tokens, 2560, generator=torch.Generator().manual_seed(20260901 + tokens)),
                torch.randn(1, 15360, generator=torch.Generator().manual_seed(20260911 + tokens)))
            expected = fixed(values[0], values[1], 'attn', 'in')
            torch.testing.assert_close(probe(*values), expected, rtol=0, atol=0)
            torch.testing.assert_close(exported_model(*values), expected, rtol=0, atol=0)
            error = _assert_close(f'checkpoint-ada-embedding independent PNNX tokens={tokens}',
                                  generated(*values), expected, 2e-5)
            python_results.append({'fixture': 'checkpoint-ada-embedding', 'tokens': tokens, 'max_abs': error})
            if tokens == 7:
                _assert_close('checkpoint-ada-embedding TorchScript PNNX baseline',
                              ts_generated(*values), expected, 2e-5)
            if runner:
                for mode in (('cpu', 'vulkan') if args.vulkan else ('cpu',)):
                    for frontend, graph in ([('pt2', directory / 'converted')] +
                                            ([('torchscript', ts_dir)] if tokens == 7 else [])):
                        input_paths = []
                        for index, value in enumerate(values):
                            input_path = directory / f'{tokens}.input{index}.f32'
                            value.numpy().tofile(input_path)
                            input_paths.append(input_path)
                        output_path = directory / f'{frontend}-{tokens}-{mode}.output.f32'
                        prefix = [str(repo / 'tools/with-nvidia-vulkan-runtime.sh')] if mode == 'vulkan' else []
                        native_command = prefix + [str(runner), mode,
                            str(graph / (stem + '.ncnn.param')), str(graph / (stem + '.ncnn.bin')), '--multi', '2',
                            'in0', str(input_paths[0]), '2560', str(tokens),
                            'in1', str(input_paths[1]), '15360', '1',
                            'out0', str(output_path)]
                        started = time.perf_counter()
                        subprocess.run(native_command, check=True)
                        elapsed = time.perf_counter() - started
                        actual = torch.from_numpy(np.fromfile(output_path, dtype=np.float32)).reshape(expected.shape)
                        error = _assert_close(f'checkpoint-ada-embedding {frontend} ncnn {mode} tokens={tokens}',
                                              actual, expected, 2e-4)
                        native.append({'fixture': 'checkpoint-ada-embedding', 'frontend': frontend, 'mode': mode,
                                       'tokens': tokens, 'max_abs': error,
                                       'process_wall_seconds': elapsed, 'command': native_command})
    fixtures.append(json.loads((directory / (stem + '.manifest.json')).read_text()))


def run_ada_mlp_embedding_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures):
    """Validate the checkpoint-backed MLP Ada branch from packed embedding."""
    directory = work / 'checkpoint-ada-mlp-embedding'
    directory.mkdir(exist_ok=True)
    command = [sys.executable, str(repo / 'tools/export_pt2_seedvr2.py'),
               '--model', 'checkpoint-ada-mlp-embedding', '--format', 'both', '--dynamic',
               '--checkpoint', str(args.checkpoint_mlp.resolve()), '--output-dir', str(directory)]
    subprocess.run(command, check=True)
    stem = 'dit_block0_video_ada_mlp_embedding'
    model_pt2 = directory / (stem + '.pt2')
    record = convert(model_pt2, pnnx, directory / 'converted', overwrite=True)
    contract = record['frontend_contract']
    assert all(operator in contract['operators'] for operator in
               ('aten.rms_norm.default', 'aten.reshape.default', 'aten.slice.Tensor',
                'aten.select.int', 'aten.linear.default', 'aten.silu.default',
                'aten.add.Tensor', 'aten.mul.Tensor'))
    assert len(contract['inputs']) == 2 and len(contract['outputs']) == 1
    reference = torch.load(directory / (stem + '_reference.pt'), weights_only=True)
    eager = load_checkpoint_video_ada_mlp(args.checkpoint_mlp)[0]
    expected_state = eager.state_dict()
    exported = torch.export.load(model_pt2)
    assert set(exported.state_dict) == set(expected_state)
    for key, value in exported.state_dict.items():
        torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
        entry = next(v for v in contract['weights'].values() if v['target'] == key)
        assert entry['sha256'] == hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()

    # Keep the numerical oracle independent from the exporter module: this is
    # the reference FixedAdaBranch + FixedSwiGLU ordering used by the block.
    reference_norm = FixedRMSNorm(2560, eps=1e-5, elementwise_affine=False).eval()
    reference_mlp = FixedSwiGLU(2560, expand_ratio=4).eval()
    reference_mlp.load_state_dict({key[4:]: value for key, value in expected_state.items()
                                   if key.startswith('mlp.')})
    reference_ada = FixedAdaBranch(2560).eval()
    with torch.no_grad():
        for name in ('shift', 'scale', 'gate'):
            getattr(reference_ada, f'mlp_{name}').copy_(expected_state[f'mlp_{name}'])

    def reference_forward(tokens, embedding):
        hidden = reference_norm(tokens)
        hidden = reference_ada(hidden, embedding, 'mlp', 'in')
        hidden = reference_mlp(hidden)
        return tokens + reference_ada(hidden, embedding, 'mlp', 'out')

    generated = _load_generated(directory / 'converted' / (stem + '_pnnx.py'))

    ts_dir = directory / 'torchscript'
    ts_dir.mkdir(exist_ok=True)
    ts = ts_dir / (stem + '.pt')
    shutil.copyfile(directory / (stem + '.pt'), ts)
    subprocess.run([str(pnnx), str(ts), 'inputshape=[7,2560],[1,15360]', 'fp16=0'],
                   check=True, cwd=ts_dir)
    ts_generated = _load_generated(ts_dir / (stem + '_pnnx.py'))
    validate_native_layers(ts_dir / (stem + '.ncnn.param'))
    with torch.no_grad():
        for tokens in (7, 13):
            values = reference['inputs'] if tokens == 7 else (
                torch.randn(tokens, 2560, generator=torch.Generator().manual_seed(20261001 + tokens)),
                torch.randn(1, 15360, generator=torch.Generator().manual_seed(20261011 + tokens)))
            expected = reference_forward(*values)
            torch.testing.assert_close(eager(*values), expected, rtol=0, atol=0)
            torch.testing.assert_close(exported.module()(*values), expected, rtol=0, atol=0)
            error = _assert_close(f'checkpoint-ada-mlp-embedding independent PNNX tokens={tokens}',
                                  generated(*values), expected, 2e-5)
            python_results.append({'fixture': 'checkpoint-ada-mlp-embedding', 'tokens': tokens,
                                   'max_abs': error})
            if tokens == 7:
                _assert_close('checkpoint-ada-mlp-embedding TorchScript PNNX baseline',
                              ts_generated(*values), expected, 2e-5)
            if runner:
                for mode in (('cpu', 'vulkan') if args.vulkan else ('cpu',)):
                    for frontend, graph in ([('pt2', directory / 'converted')] +
                                            ([('torchscript', ts_dir)] if tokens == 7 else [])):
                        input_paths = []
                        for index, value in enumerate(values):
                            input_path = directory / f'{tokens}.input{index}.f32'
                            value.numpy().tofile(input_path)
                            input_paths.append(input_path)
                        output_path = directory / f'{frontend}-{tokens}-{mode}.output.f32'
                        prefix = [str(repo / 'tools/with-nvidia-vulkan-runtime.sh')] if mode == 'vulkan' else []
                        native_command = prefix + [str(runner), mode,
                            str(graph / (stem + '.ncnn.param')), str(graph / (stem + '.ncnn.bin')), '--multi', '2',
                            'in0', str(input_paths[0]), '2560', str(tokens),
                            'in1', str(input_paths[1]), '15360', '1',
                            'out0', str(output_path)]
                        started = time.perf_counter()
                        subprocess.run(native_command, check=True)
                        elapsed = time.perf_counter() - started
                        actual = torch.from_numpy(np.fromfile(output_path, dtype=np.float32)).reshape(expected.shape)
                        error = _assert_close(f'checkpoint-ada-mlp-embedding {frontend} ncnn {mode} tokens={tokens}',
                                              actual, expected, 2e-4)
                        native.append({'fixture': 'checkpoint-ada-mlp-embedding', 'frontend': frontend,
                                       'mode': mode, 'tokens': tokens, 'max_abs': error,
                                       'process_wall_seconds': elapsed, 'command': native_command})
    fixtures.append(json.loads((directory / (stem + '.manifest.json')).read_text()))


def run_ada_attention_gate_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures):
    """Validate packed attention gating and the residual using checkpoint state."""
    directory = work / 'checkpoint-ada-attn-gate-embedding'
    directory.mkdir(exist_ok=True)
    command = [sys.executable, str(repo / 'tools/export_pt2_seedvr2.py'),
               '--model', 'checkpoint-ada-attn-gate-embedding', '--format', 'both', '--dynamic',
               '--checkpoint', str(args.checkpoint_mlp.resolve()), '--output-dir', str(directory)]
    subprocess.run(command, check=True)
    stem = 'dit_block0_video_ada_attn_gate_embedding'
    model_pt2 = directory / (stem + '.pt2')
    record = convert(model_pt2, pnnx, directory / 'converted', overwrite=True)
    contract = record['frontend_contract']
    assert all(operator in contract['operators'] for operator in
               ('aten.reshape.default', 'aten.slice.Tensor', 'aten.select.int',
                'aten.add.Tensor', 'aten.mul.Tensor'))
    assert len(contract['inputs']) == 3 and len(contract['outputs']) == 1
    reference = torch.load(directory / (stem + '_reference.pt'), weights_only=True)
    eager, checkpoint_sha256 = load_checkpoint_video_ada_attention_gate(args.checkpoint_mlp)
    ada = FixedAdaBranch(2560).eval()
    with torch.no_grad():
        ada.attn_gate.copy_(eager.attn_gate)
    expected_state = eager.state_dict()
    exported = torch.export.load(model_pt2)
    assert set(exported.state_dict) == set(expected_state)
    for key, value in exported.state_dict.items():
        torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
        entry = next(v for v in contract['weights'].values() if v['target'] == key)
        assert entry['sha256'] == hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
    assert json.loads((directory / (stem + '.manifest.json')).read_text())['checkpoint_sha256'] == checkpoint_sha256
    generated = _load_generated(directory / 'converted' / (stem + '_pnnx.py'))

    ts_dir = directory / 'torchscript'
    ts_dir.mkdir(exist_ok=True)
    ts = ts_dir / (stem + '.pt')
    shutil.copyfile(directory / (stem + '.pt'), ts)
    subprocess.run([str(pnnx), str(ts),
                    'inputshape=[7,2560],[7,2560],[1,15360]', 'fp16=0'],
                   check=True, cwd=ts_dir)
    ts_generated = _load_generated(ts_dir / (stem + '_pnnx.py'))
    validate_native_layers(ts_dir / (stem + '.ncnn.param'))
    with torch.no_grad():
        for tokens in (7, 13):
            values = reference['inputs'] if tokens == 7 else (
                torch.randn(tokens, 2560, generator=torch.Generator().manual_seed(20261021 + tokens)),
                torch.randn(tokens, 2560, generator=torch.Generator().manual_seed(20261031 + tokens)),
                torch.randn(1, 15360, generator=torch.Generator().manual_seed(20261041 + tokens)))
            expected = eager(*values)
            oracle = values[1] + ada(values[0], values[2], 'attn', 'out')
            torch.testing.assert_close(expected, oracle, rtol=0, atol=0)
            torch.testing.assert_close(exported.module()(*values), expected, rtol=0, atol=0)
            error = _assert_close(f'checkpoint-ada-attn-gate-embedding PNNX tokens={tokens}',
                                  generated(*values), expected, 2e-5)
            python_results.append({'fixture': 'checkpoint-ada-attn-gate-embedding',
                                   'tokens': tokens, 'max_abs': error})
            if tokens == 7:
                _assert_close('checkpoint-ada-attn-gate-embedding TorchScript PNNX baseline',
                              ts_generated(*values), expected, 2e-5)
            if runner:
                for mode in (('cpu', 'vulkan') if args.vulkan else ('cpu',)):
                    for frontend, graph in ([('pt2', directory / 'converted')] +
                                            ([('torchscript', ts_dir)] if tokens == 7 else [])):
                        input_paths = []
                        for index, value in enumerate(values):
                            input_path = directory / f'{tokens}.input{index}.f32'
                            value.numpy().tofile(input_path)
                            input_paths.append(input_path)
                        output_path = directory / f'{frontend}-{tokens}-{mode}.output.f32'
                        prefix = [str(repo / 'tools/with-nvidia-vulkan-runtime.sh')] if mode == 'vulkan' else []
                        native_command = prefix + [str(runner), mode,
                            str(graph / (stem + '.ncnn.param')), str(graph / (stem + '.ncnn.bin')),
                            '--multi', '3',
                            'in0', str(input_paths[0]), '2560', str(tokens),
                            'in1', str(input_paths[1]), '2560', str(tokens),
                            'in2', str(input_paths[2]), '15360', '1',
                            'out0', str(output_path)]
                        started = time.perf_counter()
                        subprocess.run(native_command, check=True)
                        elapsed = time.perf_counter() - started
                        actual = torch.from_numpy(np.fromfile(output_path, dtype=np.float32)).reshape(expected.shape)
                        error = _assert_close(
                            f'checkpoint-ada-attn-gate-embedding {frontend} ncnn {mode} tokens={tokens}',
                            actual, expected, 2e-4)
                        native.append({'fixture': 'checkpoint-ada-attn-gate-embedding',
                                       'frontend': frontend, 'mode': mode, 'tokens': tokens,
                                       'max_abs': error, 'process_wall_seconds': elapsed,
                                       'command': native_command})
    fixtures.append(json.loads((directory / (stem + '.manifest.json')).read_text()))


def run_attention_core_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures, checkpoint=False, boundary=False):
    """Validate the fixed-shape QKV/RMSNorm/RoPE/SDPA attention core."""
    name = ('checkpoint-ada-attention' if boundary else
            'checkpoint-attention-core' if checkpoint else 'attention-core')
    directory = work / name
    directory.mkdir(exist_ok=True)
    tokens, width = 7, 2560 if checkpoint else 16
    command = [sys.executable, str(repo / 'tools/export_pt2_seedvr2.py'),
               '--model', name, '--format', 'both', '--tokens', str(tokens),
               '--output-dir', str(directory)]
    if checkpoint:
        command.extend(['--checkpoint', str(args.checkpoint_mlp.resolve())])
    subprocess.run(command, check=True)
    stem = ('dit_block0_video_ada_attention' if boundary else
            'dit_block0_video_attention_core' if checkpoint else 'seedvr2_fixed_attention_core')
    model_pt2 = directory / (stem + '.pt2')
    record = convert(model_pt2, pnnx, directory / 'converted', overwrite=True)
    contract = record['frontend_contract']
    assert all(operator in contract['operators'] for operator in
               ('aten.linear.default', 'aten.reshape.default', 'aten.select.int',
                'aten.rms_norm.default', 'aten.matmul.default', 'aten.permute.default',
                'aten.scaled_dot_product_attention.default'))
    assert len(contract['inputs']) == (2 if boundary else 1) and len(contract['outputs']) == 1
    assert not contract['range_constraints']
    reference = torch.load(directory / (stem + '_reference.pt'), weights_only=True)
    eager = (load_checkpoint_video_ada_attention(args.checkpoint_mlp, tokens)[0] if boundary else
             load_checkpoint_video_attention_core(args.checkpoint_mlp, tokens)[0] if checkpoint
             else SeedVR2AttentionCoreProbe(tokens=tokens).eval())
    eager.load_state_dict(reference['state_dict'], strict=True)
    exported = torch.export.load(model_pt2)
    assert set(exported.state_dict) == set(eager.state_dict())
    for key, value in exported.state_dict.items():
        torch.testing.assert_close(value, eager.state_dict()[key], rtol=0, atol=0)
        entry = next(v for v in contract['weights'].values() if v['target'] == key)
        assert entry['sha256'] == hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
    generated = _load_generated(directory / 'converted' / (stem + '_pnnx.py'))

    ts_dir = directory / 'torchscript'
    ts_dir.mkdir(exist_ok=True)
    ts = ts_dir / (stem + '.pt')
    shutil.copyfile(directory / (stem + '.pt'), ts)
    shape = f'inputshape=[7,{width}]' + (',[1,15360]' if boundary else '')
    subprocess.run([str(pnnx), str(ts), shape, 'fp16=0'], check=True, cwd=ts_dir)
    ts_generated = _load_generated(ts_dir / (stem + '_pnnx.py'))
    validate_native_layers(ts_dir / (stem + '.ncnn.param'))
    values = reference['inputs']
    with torch.no_grad():
        expected = eager(*values)
        # Independent oracle: scalar rotary formula and explicit softmax attention,
        # rather than the export adapter's dense rotation matrix and SDPA call.
        core = eager.attention if boundary else eager
        hidden = values[0]
        if boundary:
            ada = FixedAdaBranch(width).eval()
            for field in ('shift', 'scale', 'gate'):
                getattr(ada, 'attn_' + field).copy_(getattr(eager, 'attn_' + field))
            hidden = ada(FixedRMSNorm(width, 1e-5, False)(hidden), values[1], 'attn', 'in')
        qkv = core.proj_qkv(hidden).reshape(tokens, 3, core.heads, core.head_dim)
        q, k, v = qkv.unbind(1)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-5) * core.norm_q.weight
        k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-5) * core.norm_k.weight
        if checkpoint:
            freqs, _ = build_mmrope_frequencies((1, 1, tokens), 58, window_shape=(1, 1, tokens),
                                               device=torch.device('cpu'), dtype=torch.float32)
        else:
            inverse = 10000.0 ** (-torch.arange(0, core.head_dim, 2) / core.head_dim)
            freqs = (torch.arange(tokens)[:, None] * inverse).repeat_interleave(2, -1)
        q, k = _apply_fixed_rotary(q, freqs).transpose(0, 1), _apply_fixed_rotary(k, freqs).transpose(0, 1)
        scores = q @ k.transpose(-1, -2) / (core.head_dim ** 0.5)
        oracle = core.proj_out((scores.softmax(-1) @ v.transpose(0, 1)).transpose(0, 1).reshape(tokens, width))
        if boundary:
            oracle = values[0] + ada(oracle, values[1], 'attn', 'out')
        oracle_error = _assert_close(name + ' independent attention formula', expected, oracle, 2e-4)
        torch.testing.assert_close(exported.module()(*values), expected, rtol=0, atol=0)
        error = _assert_close(name + ' independent PNNX', generated(*values), expected, 2e-5)
        python_results.append({'fixture': name, 'tokens': tokens, 'max_abs': error, 'oracle_max_abs': oracle_error})
        _assert_close(name + ' TorchScript PNNX baseline', ts_generated(*values), expected, 2e-5)
        if runner:
            input_path = directory / 'input.f32'
            values[0].numpy().tofile(input_path)
            if boundary:
                embedding_path = directory / 'embedding.f32'
                values[1].numpy().tofile(embedding_path)
            for mode in (('cpu', 'vulkan') if args.vulkan else ('cpu',)):
                paired_outputs = {}
                for frontend, graph in (('pt2', directory / 'converted'),
                                        ('torchscript', ts_dir)):
                    output_path = directory / f'{frontend}-{mode}.output.f32'
                    prefix = [str(repo / 'tools/with-nvidia-vulkan-runtime.sh')] if mode == 'vulkan' else []
                    native_command = prefix + [str(runner), mode,
                        str(graph / (stem + '.ncnn.param')), str(graph / (stem + '.ncnn.bin'))]
                    if boundary:
                        native_command += ['--multi', '2']
                    native_command += ['in0', str(input_path), str(width), str(tokens)]
                    if boundary:
                        native_command += ['in1', str(embedding_path), '15360', '1']
                    native_command += ['out0', str(output_path)]
                    started = time.perf_counter()
                    subprocess.run(native_command, check=True)
                    elapsed = time.perf_counter() - started
                    actual = torch.from_numpy(np.fromfile(output_path, dtype=np.float32)).reshape(expected.shape)
                    paired_outputs[frontend] = actual
                    error = _assert_close(f'{name} {frontend} ncnn {mode}', actual, expected, 2e-4)
                    native.append({'fixture': name, 'frontend': frontend, 'mode': mode,
                                   'tokens': tokens, 'max_abs': error,
                                   'process_wall_seconds': elapsed, 'command': native_command})
                paired_error = _assert_close(f'{name} PT2 vs TorchScript ncnn {mode}',
                    paired_outputs['pt2'], paired_outputs['torchscript'], 2e-4)
                for result in native[-2:]:
                    result['pt2_vs_torchscript_max_abs'] = paired_error
    fixtures.append(json.loads((directory / (stem + '.manifest.json')).read_text()))


def run_full_block_export_fixture(args, repo, pnnx, runner, work, native, fixtures,
                                  block_index=0):
    """Export and validate the complete fixed-shape block conversion."""
    fixture_name = f'checkpoint-dit-block{block_index}'
    directory = work / fixture_name
    directory.mkdir(exist_ok=True)
    command = [sys.executable, str(repo / 'tools/export_pt2_seedvr2.py'),
               '--model', 'checkpoint-dit-block', '--format', 'both',
               '--block-index', str(block_index), '--checkpoint',
               str(args.checkpoint_mlp.resolve()), '--output-dir', str(directory)]
    subprocess.run(command, check=True)
    stem = f'dit_block{block_index}_full'
    model_pt2 = directory / (stem + '.pt2')
    exported = torch.export.load(model_pt2)
    reference = torch.load(directory / (stem + '_reference.pt'), weights_only=True)
    eager = load_fixed_dit_block(args.checkpoint_mlp, block_index=block_index).eval()
    torch.testing.assert_close(exported.module()(*reference['inputs']), reference['outputs'], rtol=0, atol=0)
    torch.testing.assert_close(eager(*reference['inputs']), reference['outputs'], rtol=0, atol=0)
    ts = torch.jit.load(str(directory / (stem + '.pt'))).eval()
    torch.testing.assert_close(ts(*reference['inputs']), reference['outputs'], rtol=0, atol=0)
    record = convert(model_pt2, pnnx, directory / 'converted', overwrite=True,
                     rewrite_dit_block=True)
    assert record['status'] == 'success'
    graph = directory / 'converted'
    ts_dir = directory / 'torchscript'
    ts_dir.mkdir(exist_ok=True)
    ts_path = ts_dir / (stem + '.pt')
    shutil.copyfile(directory / (stem + '.pt'), ts_path)
    subprocess.run([str(pnnx), str(ts_path),
                    'inputshape=[3600,2560],[58,2560],[1,15360]', 'fp16=0'],
                   check=True, cwd=ts_dir)
    rewrite_ncnn_param(ts_dir / (stem + '.ncnn.param'),
                       size=DIT_BLOCK_CONTRACT.source_shape,
                       windows=DIT_BLOCK_CONTRACT.window_shape,
                       text_tokens=DIT_BLOCK_CONTRACT.text_tokens)
    validate_native_layers(ts_dir / (stem + '.ncnn.param'))
    if runner:
        input_paths = []
        for index, value in enumerate(reference['inputs']):
            path = directory / f'input{index}.f32'
            value.numpy().tofile(path)
            input_paths.append(path)
        for mode in (('cpu', 'vulkan') if args.vulkan else ('cpu',)):
            for output_index, expected in enumerate(reference['outputs']):
                # The two branches contain large FP32 reductions (the video
                # branch has 3,600 tokens and the text branch has 58).  ncnn's
                # reduction order differs from PyTorch; the measured fixed
                # block-0/1/9 runs stay below 5e-4 on both outputs and
                # backends.  Keep this bound scoped to the fixed full-block
                # acceptance fixture rather than reusing it for small graphs.
                eager_tolerance = 5e-4
                paired_outputs = {}
                for frontend, frontend_graph in (('pt2', graph), ('torchscript', ts_dir)):
                    output_path = directory / f'{frontend}-{mode}-out{output_index}.f32'
                    command = ([str(repo / 'tools/with-nvidia-vulkan-runtime.sh')] if mode == 'vulkan' else [])
                    command += [str(runner), mode, str(frontend_graph / (stem + '.ncnn.param')),
                                str(frontend_graph / (stem + '.ncnn.bin')), '--multi', '3']
                    command += ['in0', str(input_paths[0]), '2560', '3600',
                                'in1', str(input_paths[1]), '2560', '58',
                                'in2', str(input_paths[2]), '15360', '1',
                                f'out{output_index}', str(output_path)]
                    started = time.perf_counter()
                    subprocess.run(command, check=True)
                    elapsed = time.perf_counter() - started
                    actual = torch.from_numpy(np.fromfile(output_path, dtype=np.float32)).reshape(expected.shape)
                    paired_outputs[frontend] = actual
                    error = _assert_close(
                        f'{fixture_name} {frontend} ncnn {mode} output={output_index}',
                        actual, expected, eager_tolerance)
                    native.append({'fixture': fixture_name, 'frontend': frontend,
                                   'mode': mode, 'output': output_index, 'max_abs': error,
                                   'eager_tolerance': eager_tolerance,
                                   'process_wall_seconds': elapsed, 'command': command})
                paired_error = _assert_close(
                    f'{fixture_name} PT2 vs TorchScript ncnn {mode} output={output_index}',
                    paired_outputs['pt2'], paired_outputs['torchscript'], 2e-4)
                for result in native[-2:]:
                    result['pt2_vs_torchscript_max_abs'] = paired_error
    manifest = json.loads((directory / (stem + '.manifest.json')).read_text())
    manifest.update({'conversion_status': 'converted_and_validated',
                     'conversion_diagnostics': record['diagnostics'],
                     'validated': 'eager, torch.export module, TorchScript, PNNX, ncnn PT2/TorchScript graphs'} )
    (directory / (stem + '.manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    fixtures.append(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pnnx', required=True, type=Path)
    parser.add_argument('--ncnn-runner', type=Path)
    parser.add_argument('--vulkan', action='store_true')
    parser.add_argument('--checkpoint-mlp', type=Path)
    parser.add_argument('--full-block-index', type=int, default=0,
                        help='checkpoint DiT block index for the complete fixed-shape fixture')
    parser.add_argument('--only-full-block', action='store_true',
                        help='run only the complete checkpoint DiT block fixture')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--result-json', required=True, type=Path)
    args = parser.parse_args()
    if args.full_block_index < 0:
        parser.error('--full-block-index must be non-negative')
    if args.only_full_block and not args.checkpoint_mlp:
        parser.error('--only-full-block requires --checkpoint-mlp')
    pnnx = args.pnnx.resolve()
    runner = args.ncnn_runner.resolve() if args.ncnn_runner else None
    if args.vulkan and not runner:
        parser.error('--vulkan requires --ncnn-runner')
    repo = Path(__file__).resolve().parents[1]
    source = next(parent for parent in pnnx.parents if (parent / 'tools/pnnx/python/pt2_to_pnnx.py').is_file())
    args.result_json.unlink(missing_ok=True)
    native, python_results, fixtures = [], [], []
    with tempfile.TemporaryDirectory(prefix='seedvr2-independent-pt2-') as temporary:
        work = args.work_dir.resolve() if args.work_dir else Path(temporary)
        work.mkdir(parents=True, exist_ok=True)
        _unsupported_programs(work, pnnx, converter=convert)
        variants = [] if args.only_full_block else [
            ('static', 'projection', False, 8), ('dynamic', 'projection', True, 8)]
        if args.checkpoint_mlp and not args.only_full_block:
            variants.extend((
                ('checkpoint', 'checkpoint-mlp', True, 2560),
                ('checkpoint-residual', 'checkpoint-mlp-residual', True, 2560),
            ))
        for name, kind, dynamic, width in variants:
            directory = work / name
            directory.mkdir(exist_ok=True)
            command = [sys.executable, str(repo / 'tools/export_pt2_seedvr2.py'),
                       '--model', kind, '--format', 'both', '--output-dir', str(directory)]
            if dynamic:
                command.append('--dynamic')
            if kind in ('checkpoint-mlp', 'checkpoint-mlp-residual'):
                command.extend(['--checkpoint', str(args.checkpoint_mlp.resolve())])
            subprocess.run(command, check=True)
            stem = {
                'checkpoint-mlp': 'dit_block0_video_mlp',
                'checkpoint-mlp-residual': 'dit_block0_video_mlp_residual',
                'projection': 'dit_mlp_probe',
            }[kind]
            model_pt2 = directory / (stem + '.pt2')
            if name == 'static':
                _conversion_failure_contracts(work, model_pt2, converter=convert, pnnx=pnnx)
            record = convert(model_pt2, pnnx, directory / 'converted', overwrite=True)
            assert record['frontend'] == 'independent'
            contract = record['frontend_contract']
            reference = torch.load(directory / (stem + '_reference.pt'), weights_only=True)
            if kind == 'checkpoint-mlp':
                eager = load_checkpoint_video_mlp(args.checkpoint_mlp)[0]
            elif kind == 'checkpoint-mlp-residual':
                eager = load_checkpoint_video_mlp_residual(args.checkpoint_mlp)[0]
            else:
                eager = SeedVR2MLPProbe(width).eval()
            expected_state = reference.get('state_dict', eager.state_dict())
            eager.load_state_dict(expected_state)
            exported = torch.export.load(model_pt2)
            assert set(exported.state_dict) == set(expected_state)
            for key, value in exported.state_dict.items():
                torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
                entry = next(v for v in contract['weights'].values() if v['target'] == key)
                assert entry['sha256'] == hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
            assert len(contract['inputs']) == len(contract['outputs']) == 1
            assert bool(contract['range_constraints']) == dynamic
            if kind == 'checkpoint-mlp-residual':
                assert 'aten.rms_norm.default' in contract['operators']
                assert 'aten.add.Tensor' in contract['operators']
            generated = _load_generated(directory / 'converted' / (stem + '_pnnx.py'))
            ts_dir = directory / 'torchscript'
            ts_dir.mkdir(exist_ok=True)
            ts = ts_dir / (stem + '.pt')
            shutil.copyfile(directory / (stem + '.pt'), ts)
            subprocess.run([str(pnnx), str(ts), f'inputshape=[7,{width}]', 'fp16=0'], check=True, cwd=ts_dir)
            ts_generated = _load_generated(ts_dir / (stem + '_pnnx.py'))
            validate_native_layers(ts_dir / (stem + '.ncnn.param'))
            with torch.no_grad():
                for tokens in ((1, 7, 13, 32) if dynamic and kind not in ('checkpoint-mlp', 'checkpoint-mlp-residual') else ((7, 13) if dynamic else (7,))):
                    value = reference['inputs'][0] if tokens == 7 else torch.randn(
                        tokens, width, generator=torch.Generator().manual_seed(20260823 + tokens))
                    expected = eager(value)
                    torch.testing.assert_close(exported.module()(value), expected, rtol=0, atol=0)
                    error = _assert_close(f'{name} independent PNNX tokens={tokens}', generated(value), expected, 2e-5)
                    python_results.append({'fixture': name, 'tokens': tokens, 'max_abs': error})
                    if tokens == 7:
                        _assert_close(f'{name} TorchScript PNNX baseline', ts_generated(value), expected, 2e-5)
                    if runner:
                        for mode in (('cpu', 'vulkan') if args.vulkan else ('cpu',)):
                            for frontend, graph in [('pt2', directory / 'converted')] + ([('torchscript', ts_dir)] if tokens == 7 else []):
                                input_path = directory / f'{tokens}.input.f32'
                                output_path = directory / f'{frontend}-{tokens}-{mode}.output.f32'
                                value.numpy().tofile(input_path)
                                prefix = [str(repo / 'tools/with-nvidia-vulkan-runtime.sh')] if mode == 'vulkan' else []
                                native_command = prefix + [str(runner), mode, str(graph / (stem + '.ncnn.param')),
                                    str(graph / (stem + '.ncnn.bin')), 'in0', str(input_path), str(width), str(tokens), 'out0', str(output_path)]
                                started = time.perf_counter()
                                subprocess.run(native_command, check=True)
                                elapsed = time.perf_counter() - started
                                actual = torch.from_numpy(np.fromfile(output_path, dtype=np.float32)).reshape(expected.shape)
                                error = _assert_close(f'{name} {frontend} ncnn {mode} tokens={tokens}', actual, expected, 2e-4 if width > 8 else 1e-5)
                                native.append({'fixture': name, 'frontend': frontend, 'mode': mode, 'tokens': tokens,
                                    'max_abs': error, 'process_wall_seconds': elapsed, 'command': native_command})
            fixtures.append(json.loads((directory / (stem + '.manifest.json')).read_text()))
        if args.checkpoint_mlp:
            if not args.only_full_block:
                run_ada_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures)
                run_ada_embedding_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures)
                run_ada_mlp_embedding_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures)
                run_ada_attention_gate_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures)
                run_attention_core_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures, checkpoint=True)
                run_attention_core_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures, checkpoint=True, boundary=True)
            run_full_block_export_fixture(args, repo, pnnx, runner, work, native, fixtures,
                                          block_index=args.full_block_index)
        if not args.only_full_block:
            run_attention_core_fixture(args, repo, pnnx, runner, work, native, python_results, fixtures)
        def commit(path):
            return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        def dirty(path):
            return bool(subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip())
        result = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'frontend': 'independent',
            'seedvr2_commit': commit(repo), 'seedvr2_dirty': dirty(repo), 'pnnx_commit': commit(source),
            'pnnx_dirty': dirty(source), 'ncnn_commit': commit(repo / 'third_party/ncnn'),
            'pnnx_binary_sha256': sha(pnnx), 'frontend_sha256': sha(source / 'tools/pnnx/python/pt2_to_pnnx.py'),
            'patch_sha256': sha(repo / 'patches/pnnx-pt2-independent.patch'),
            'python': platform.python_version(), 'pytorch': torch.__version__, 'system': platform.platform(),
            'cpu': subprocess.check_output(['lscpu'], text=True),
            'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                '--format=csv,noheader'], text=True).splitlines() if args.vulkan else None,
            'precision': 'FP32; ncnn FP16/BF16 disabled', 'fixtures': fixtures, 'python_results': python_results,
            'native': native, 'peak_memory': None,
            'timing_scope': 'single subprocess wall time: startup + load + inference + write; no warmup'}
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Independent PT2 regression passed: {len(python_results)} Python and {len(native)} native comparisons')


if __name__ == '__main__':
    main()
