import hashlib
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from analysis.artifact_config import canonical_outputs, load_config
from scripts.verify_results import verify_results_checksums


class ResultsChecksumsTests(unittest.TestCase):
    def test_reviewed_results_match_checksum_manifest(self):
        self.assertEqual(
            [],
            verify_results_checksums(
                ROOT / "results", canonical_outputs(load_config())
            ),
        )

    def test_modified_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            product = results / "product.txt"
            product.write_text("reviewed\n", encoding="utf-8")
            digest = hashlib.sha256(product.read_bytes()).hexdigest()
            (results / "checksums.sha256").write_text(
                f"{digest}  product.txt\n", encoding="utf-8"
            )
            product.write_text("changed\n", encoding="utf-8")

            self.assertEqual(
                ["reviewed-results checksum mismatch: product.txt"],
                verify_results_checksums(results, ["product.txt"]),
            )

    def test_manifest_and_result_inventory_must_be_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            extra = results / "extra.txt"
            extra.write_text("extra\n", encoding="utf-8")
            digest = hashlib.sha256(extra.read_bytes()).hexdigest()
            (results / "checksums.sha256").write_text(
                f"{digest}  extra.txt\n", encoding="utf-8"
            )

            errors = verify_results_checksums(results, ["expected.txt"])

        self.assertEqual(
            [
                "missing reviewed-results checksum entry: expected.txt",
                "unexpected reviewed-results checksum entry: extra.txt",
                "missing reviewed result: expected.txt",
                "unexpected reviewed result: extra.txt",
            ],
            errors,
        )


if __name__ == "__main__":
    unittest.main()
