"""Storage geometry contracts for the opt-in V4.1 TP4 package."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture
def exporter():
    spec = importlib.util.spec_from_file_location(
        "_v41_exact_export", ROOT / "python/tools/aot/export_b12x_slices_aot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_exact_tp4_is_explicit_and_other_roles_keep_their_geometry(exporter):
    assert exporter.role_geometry("v41", "spark") == (384, 576, 640, 6)
    assert exporter.role_geometry("v41", "spark", exact_v41_slices=True) == (384, 576, 576, 6)
    for role, intermediate in [("spark_tp2", 1152), ("spark_tp3", 768), ("spark_tp6", 384)]:
        assert exporter.role_geometry("v41", role) == (384, intermediate, intermediate, 6)
        with pytest.raises(ValueError, match="Spark TP4"):
            exporter.role_geometry("v41", role, exact_v41_slices=True)
    with pytest.raises(ValueError, match="Spark TP4"):
        exporter.role_geometry("dsv4f", "spark", exact_v41_slices=True)
