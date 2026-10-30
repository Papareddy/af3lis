"""Config parsing: no key may be stored where nothing reads it.

The bug this guards: every config file wrote `infer_gpu_gres`, but the
sbatch renderer reads `gpu_gres`, and the flat parser stored the unknown key
under its own name -- so the configured value never reached the #SBATCH line
and the built-in default always won. Same for `align_extra` /
`af3_data_extra` and `infer_extra` / `af3_infer_extra`.

Run:  python -m pytest tests/test_config.py -q   (or plain `python tests/test_config.py`)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import pipeline  # noqa: E402


def _cfg_file(text: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "config.yaml")
    with open(p, "w") as fh:
        fh.write(text)
    return p


def _render(cfg: dict) -> dict[str, str]:
    """Render every sbatch for a one-pair build; return {name: text}."""
    d = tempfile.mkdtemp()
    jp = os.path.join(d, "bait___preyA.json")
    open(jp, "w").write("{}")
    pipeline.render_sbatches([("bait___preyA", jp)], d, "t", cfg)
    out = {}
    for n in ("align", "infer", "pack_infer", "pack_bridge"):
        out[n] = open(os.path.join(d, f"{n}.sbatch")).read()
    return out


def test_legacy_keys_reach_the_sbatch_lines():
    p = _cfg_file(
        "infer_gpu_gres: gpu:7\n"
        "align_extra: --max_template_date=2024-01-01\n"
        "infer_extra: --save_embeddings=false\n")
    cfg, warns = pipeline.parse_config(p)
    assert cfg["gpu_gres"] == "gpu:7", cfg["gpu_gres"]
    assert cfg["af3_data_extra"] == "--max_template_date=2024-01-01"
    assert cfg["af3_infer_extra"] == "--save_embeddings=false"
    # stored under the canonical name only -- never under the dead one
    for dead in ("infer_gpu_gres", "align_extra", "infer_extra"):
        assert dead not in cfg, dead
    # each alias earns a deprecation note
    for dead in ("infer_gpu_gres", "align_extra", "infer_extra"):
        assert any(f"'{dead}' is a deprecated alias" in w for w in warns), warns
    # and the value actually arrives where SLURM reads it
    sb = _render(cfg)
    assert "#SBATCH --gres=gpu:7" in sb["infer"], "infer.sbatch ignored gres"
    assert "#SBATCH --gres=gpu:7" in sb["pack_infer"], "pack_infer ignored gres"
    assert "--max_template_date=2024-01-01" in sb["align"]
    assert "--save_embeddings=false" in sb["infer"]


def test_canonical_spelling_wins_over_alias_in_either_order():
    for text in ("infer_gpu_gres: gpu:7\ngpu_gres: gpu:1\n",
                 "gpu_gres: gpu:1\ninfer_gpu_gres: gpu:7\n"):
        cfg, warns = pipeline.parse_config(_cfg_file(text))
        assert cfg["gpu_gres"] == "gpu:1", (text, cfg["gpu_gres"])
        assert any("disagree" in w for w in warns), (text, warns)


def test_unknown_key_is_warned_with_a_suggestion_and_not_stored():
    cfg, warns = pipeline.parse_config(_cfg_file("infer_gres: gpu:2\n"))
    assert "infer_gres" not in cfg
    assert cfg["gpu_gres"] == pipeline._DEFAULT_CFG["gpu_gres"]
    w = [x for x in warns if "infer_gres" in x]
    assert w and "IGNORED" in w[0], warns
    assert "did you mean 'gpu_gres'" in w[0], w[0]


def test_retired_key_gets_a_note_not_a_typo_warning():
    cfg, warns = pipeline.parse_config(_cfg_file("infer_gpumem: 40G\n"))
    assert "infer_gpumem" not in cfg
    assert len(warns) == 1 and "is ignored" in warns[0], warns
    assert "did you mean" not in warns[0]


def test_nested_sections_accept_canonical_and_legacy_children():
    cfg, warns = pipeline.parse_config(_cfg_file(
        "infer:\n"
        "  partition: gpu-test\n"
        "  infer_gpu_gres: gpu:3\n"
        "  bogus_child: 1\n"
        "align:\n"
        "  af3_data_extra: --x\n"))
    assert cfg["infer_partition"] == "gpu-test"
    assert cfg["gpu_gres"] == "gpu:3"
    assert cfg["af3_data_extra"] == "--x"
    assert "bogus_child" not in cfg
    assert any("infer.bogus_child" in w and "IGNORED" in w for w in warns), warns


def test_default_gres_never_names_a_card():
    assert pipeline._DEFAULT_CFG["gpu_gres"] == "gpu:1"
    cfg, warns = pipeline.parse_config(None)
    assert warns == []
    sb = _render(cfg)
    assert "#SBATCH --gres=gpu:1\n" in sb["infer"]
    assert "#SBATCH --gres=gpu:1\n" in sb["pack_infer"]


def test_pinned_gres_is_honoured_but_flagged():
    cfg, warns = pipeline.parse_config(_cfg_file("gpu_gres: gpu:A100:1\n"))
    assert cfg["gpu_gres"] == "gpu:A100:1"
    assert any("pins a card type" in w for w in warns), warns


def test_example_config_uses_only_keys_that_are_read():
    p = os.path.join(HERE, "config.example.yaml")
    cfg, warns = pipeline.parse_config(p)
    assert warns == [], warns
    text = open(p).read()
    for dead in ("infer_gpu_gres:", "align_extra:", "infer_extra:",
                 "infer_gpumem:"):
        assert f"\n{dead}" not in text, f"example still writes {dead}"


def test_missing_explicit_config_is_reported():
    d = tempfile.mkdtemp()
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "pipeline.py"),
         "--config", os.path.join(d, "nope.yaml"),
         "--chains-tsv-template", os.path.join(d, "c.tsv")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "does not exist" in r.stderr, r.stderr


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
