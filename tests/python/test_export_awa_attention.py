import sys
from pathlib import Path

import tools.export_awa_attention as exporter


def test_pnnx_export_uses_an_absolute_model_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "converted"
    pnnx = tmp_path / "pnnx"
    pnnx.touch()
    invocation = {}

    def fake_run(command, *, cwd, check):
        invocation["command"] = command
        invocation["cwd"] = cwd
        invocation["check"] = check

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_awa_attention.py",
            "--output-dir",
            "converted",
            "--pnnx",
            str(pnnx),
        ],
    )

    exporter.main()

    assert Path(invocation["command"][1]) == (output_dir / "awa_attention.pt").resolve()
    assert Path(invocation["cwd"]) == output_dir.resolve()
    assert invocation["check"] is True
