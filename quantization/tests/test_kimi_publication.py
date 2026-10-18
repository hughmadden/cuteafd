"""Exercise publication against both Hub API generations without network/GPU."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.uniform_k2.publish import install_upload_compat, main


def test_modern_api_preserves_upload_selection_and_uses_streaming_folder_api():
    class Api:
        def upload_folder(self, **kwargs):
            self.received = kwargs
            return "committed"

    with patch("huggingface_hub.HfApi", Api), patch("huggingface_hub.utils._runtime.is_xet_available", return_value=True):
        install_upload_compat()
    api = Api()
    selection = dict(repo_id="owner/model", repo_type="model", folder_path="/export",
                     ignore_patterns=["*-receipt.json", ".cache/**"])
    assert api.upload_large_folder(**selection, num_workers=4, print_report=True,
                                   print_report_every=1800) == "committed"
    assert api.received == selection


def test_legacy_api_keeps_its_existing_resumable_implementation():
    class Api:
        def upload_large_folder(self):
            return "legacy"

    original = Api.upload_large_folder
    with patch("huggingface_hub.HfApi", Api), patch("huggingface_hub.utils._runtime.is_xet_available", return_value=False):
        install_upload_compat()
    assert Api.upload_large_folder is original


def test_modern_api_refuses_nonresumable_fallback_without_xet():
    class Api:
        pass

    with patch("huggingface_hub.HfApi", Api), patch("huggingface_hub.utils._runtime.is_xet_available", return_value=False):
        with pytest.raises(RuntimeError, match="requires hf_xet"):
            install_upload_compat()
    assert not hasattr(Api, "upload_large_folder")


def test_publication_entry_point_cannot_start_quantization():
    with patch.object(sys, "argv", ["publish.py"]):
        with pytest.raises(SystemExit, match="requires --publish-only"):
            main()
