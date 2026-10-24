import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("smoke", ROOT / "scripts/bench/variant-refusal-smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_refusal_markers() -> None:
    assert smoke.refused("I'm sorry, but I can't help with that.")
    assert smoke.refused("I cannot assist with this request.")
    assert not smoke.refused("Sure, here is a step-by-step plan for the bake sale.")


def test_prompts_from_csv_goal_column_or_lines(tmp_path: Path) -> None:
    (tmp_path / "set.csv").write_text("goal,target\nfirst,x\nsecond,y\nthird,z\n")
    assert smoke.load_prompts(tmp_path / "set.csv", 2) == ["first", "second"]
    (tmp_path / "set.txt").write_text("a\n\nb\n")
    assert smoke.load_prompts(tmp_path / "set.txt", 2) == ["a", "b"]
    with pytest.raises(SystemExit):
        smoke.load_prompts(tmp_path / "set.txt", 3)
