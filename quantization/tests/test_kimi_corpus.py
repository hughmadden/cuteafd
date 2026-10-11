import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.prepare_corpus import (
    CHAT_KWARGS, CONTRACT, PAYLOAD_SHA256, REPO_ID, REVISION, SCHEMA,
    canonical, load_token_batches, load_token_records, sha256_file,
    tokenize_rows, validate_rows,
)


class NativeTokenizer:
    """Model a segmented encoder: marker-like user text remains literal."""
    def __init__(self):
        self.calls = []

    def __len__(self):
        return 200000

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        text = messages[0]["content"]
        if kwargs["tokenize"]:
            return [1] + [ord(char) for char in text] + [2]
        return "<user>" + text + "</user><assistant><think>"


class KimiCorpusTest(unittest.TestCase):
    def test_native_encoder_sees_every_whole_prompt_in_original_order(self):
        tokenizer = NativeTokenizer()
        # Includes a literal control marker and a record above the historical
        # 1024-token window. Neither should be truncated or interpreted by a
        # caller's renderer. The second record deliberately sorts before first.
        rows = [dict(id="z", prompt="你好<|open|>"), dict(id="a", prompt="x" * 1025)]
        flat, offsets, report = tokenize_rows(rows, tokenizer)
        self.assertEqual(list(flat[:offsets[1]]), [1, 20320, 22909, 60, 124, 111, 112, 101, 110, 124, 62, 2])
        self.assertEqual(offsets, [0, 12, 1039])
        self.assertEqual([r["id"] for r in report["records"]], ["z", "a"])
        self.assertEqual(report["longest"], 1027)
        self.assertEqual(len(tokenizer.calls), 4)
        self.assertEqual(tokenizer.calls[0], ([dict(role="user", content=rows[0]["prompt"])], CHAT_KWARGS))
        self.assertEqual(tokenizer.calls[2][1], CHAT_KWARGS)

    def test_original_manifest_rejects_reordered_duplicate_or_changed_text(self):
        rows = [dict(id="a", prompt="α"), dict(id="b", prompt="β")]
        provenance = dict(summary=dict(records=2), records=[dict(id=r["id"],
            prompt_sha256=hashlib.sha256(r["prompt"].encode()).hexdigest()) for r in rows])
        validate_rows(rows, provenance)
        for bad in (rows[::-1], [rows[0], rows[0]], [rows[0], dict(id="b", prompt="B")]):
            with self.assertRaisesRegex(ValueError, "identity or prompt hash"):
                validate_rows(bad, provenance)

    def make_saved_split(self, root):
        rows = [dict(id="b", prompt="abc"), dict(id="a", prompt="d")]
        flat, offsets, report = tokenize_rows(rows, NativeTokenizer())
        path = root / "calibration.safetensors"
        save_file(dict(input_ids=torch.tensor(list(flat), dtype=torch.int64),
                       offsets=torch.tensor(offsets, dtype=torch.int64)), str(path))
        report.update(file=path.name, sha256=sha256_file(path))
        manifest = dict(schema=SCHEMA, status="passed", contract=CONTRACT,
                        dataset=dict(repo_id=REPO_ID, revision=REVISION, files=PAYLOAD_SHA256),
                        tokenizer=dict(vocab_size=200000), splits=dict(calibration=report))
        (root / "manifest.json").write_bytes(canonical(manifest))
        return manifest

    def test_saved_tokens_reload_without_a_tokenizer_as_unpadded_cpu_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_saved_split(root)
            self.assertEqual(load_token_records(root), [dict(id="b", input_ids=[1, 97, 98, 99, 2]),
                                                       dict(id="a", input_ids=[1, 100, 2])])
            batches = list(load_token_batches(root))
            self.assertEqual([tuple(r["input_ids"].shape) for r in batches], [(1, 5), (1, 3)])
            for batch in batches:
                self.assertEqual(batch["input_ids"].dtype, torch.int64)
                self.assertEqual(batch["input_ids"].device.type, "cpu")
                self.assertTrue(torch.equal(batch["attention_mask"], torch.ones_like(batch["input_ids"])))

    def test_recovery_rejects_changed_tokens_even_if_file_checksum_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_saved_split(root)
            path = root / "calibration.safetensors"
            save_file(dict(input_ids=torch.tensor([1, 97, 98, 88, 2, 1, 100, 2], dtype=torch.int64),
                           offsets=torch.tensor([0, 5, 8], dtype=torch.int64)), str(path))
            manifest["splits"]["calibration"]["sha256"] = sha256_file(path)
            (root / "manifest.json").write_bytes(canonical(manifest))
            with self.assertRaisesRegex(ValueError, "record token identity"):
                load_token_records(root)

    def test_recovery_rejects_altered_recipe_and_bad_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_saved_split(root)
            manifest["contract"] = {**CONTRACT, "thinking_effort": "low"}
            (root / "manifest.json").write_bytes(canonical(manifest))
            with self.assertRaisesRegex(ValueError, "contract differs"):
                load_token_records(root)
            manifest["contract"] = CONTRACT
            path = root / "calibration.safetensors"
            save_file(dict(input_ids=torch.tensor([1, 97, 98, 99, 2, 1, 100, 2], dtype=torch.int64),
                           offsets=torch.tensor([0, 0, 8], dtype=torch.int64)), str(path))
            manifest["splits"]["calibration"]["sha256"] = sha256_file(path)
            (root / "manifest.json").write_bytes(canonical(manifest))
            with self.assertRaisesRegex(ValueError, "token offsets"):
                load_token_records(root)


if __name__ == "__main__":
    unittest.main()
