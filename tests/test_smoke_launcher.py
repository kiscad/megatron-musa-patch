"""Exercise launcher path handling without importing torch or starting training."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "examples/run_pretrain_smoke.sh"


def _launch(tmp_path, output=None):
    upstream = tmp_path / "upstream"
    upstream.mkdir(exist_ok=True)
    interpreter = tmp_path / "record-args"
    interpreter.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$MMP_CAPTURE"\n')
    interpreter.chmod(0o755)
    env = dict(
        os.environ,
        MEGATRON_LM_PATH=str(upstream),
        PYTHON=str(interpreter),
        MMP_CAPTURE=str(tmp_path / "args"),
        TMPDIR=str(tmp_path),
    )
    env.pop("OUTPUT_DIR", None)
    if output is not None:
        env["OUTPUT_DIR"] = str(output)
    return subprocess.run(
        ["bash", str(SCRIPT)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10
    )


def test_smoke_preserves_existing_output(tmp_path):
    output = tmp_path / "checkpoints"
    output.mkdir()
    checkpoint = output / ".existing-checkpoint"
    checkpoint.write_text("keep")
    result = _launch(tmp_path, output)
    assert result.returncode != 0
    assert "must be new or empty" in result.stderr
    assert checkpoint.read_text() == "keep"
    assert not (tmp_path / "args").exists()


def test_smoke_resolves_relative_output_before_chdir(tmp_path):
    result = _launch(tmp_path, "run output")
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "args").read_text().splitlines()
    assert args[args.index("--save") + 1] == str(tmp_path / "run output")


def test_smoke_uses_unique_default_output(tmp_path):
    outputs = []
    for _ in range(2):
        result = _launch(tmp_path)
        assert result.returncode == 0, result.stderr
        args = (tmp_path / "args").read_text().splitlines()
        outputs.append(Path(args[args.index("--save") + 1]))
    assert outputs[0] != outputs[1]
    assert all(path.is_dir() and path.parent == tmp_path for path in outputs)
