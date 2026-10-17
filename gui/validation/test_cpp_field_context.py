"""Validate owned C++ contexts with test-only reconstruction and frozen v1 data.

Set RCWA_FIELD_CONTEXT_EXE to the built test_rcwa_field_context executable. No sampler
is exposed in the public C++ API, and this test never regenerates the corpus.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np

REFERENCE = Path(__file__).resolve().parent / "field_reference"


def complex_array(pairs):
    values = np.asarray(pairs, dtype=np.float64)
    return values[..., 0] + 1j * values[..., 1]


@unittest.skipUnless(os.environ.get("RCWA_FIELD_CONTEXT_EXE"), "set RCWA_FIELD_CONTEXT_EXE to the C++ context test executable")
class CppFieldContextReference(unittest.TestCase):
    def test_six_harmonics_and_reconstructed_complex_fields(self):
        manifest_path = REFERENCE / "reference_v1.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive_path = REFERENCE / manifest["archive"]["filename"]
        self.assertEqual(hashlib.sha256(archive_path.read_bytes()).hexdigest(), manifest["archive"]["sha256"])
        completed = subprocess.run([os.environ["RCWA_FIELD_CONTEXT_EXE"], "--reference", str(manifest_path)],
                                   check=True, capture_output=True, text=True, timeout=90)
        actual = json.loads(completed.stdout)
        self.assertEqual([value["id"] for value in actual], [record["parameters"]["id"] for record in manifest["cases"]])
        with np.load(archive_path, allow_pickle=False) as baseline:
            for value, record in zip(actual, manifest["cases"]):
                spec = record["parameters"]
                case = spec["id"]
                with self.subTest(case=case):
                    self.assertEqual(value["n_zero_modes"], spec["expected_zero_modes"])
                    self.assertEqual(value["n_schur"], spec["expected_schur_layers"])
                    harmonics = complex_array(value["harmonics"])
                    self.assertTrue(np.isfinite(harmonics).all())
                    np.testing.assert_allclose(harmonics, baseline[case + "__harmonics"], **spec["tolerance"])
                    for source, key in (("kxm", "kx"), ("kzI", "kzI"), ("kzII", "kzII")):
                        np.testing.assert_allclose(complex_array(value[source]), baseline[case + "__" + key], **spec["tolerance"])
                    np.testing.assert_allclose(value["ky"], baseline[case + "__ky"], **spec["tolerance"])
                    np.testing.assert_allclose(value["incident_amplitudes"], baseline[case + "__incident_amplitudes"], atol=0, rtol=0)
                    carrier = np.exp(-1j * (np.outer(baseline[case + "__x"], complex_array(value["kxm"])) + value["ky"] * spec["y"]))
                    fields = np.einsum("czo,xo->czx", harmonics, carrier)
                    np.testing.assert_allclose(fields, baseline[case + "__fields"], **spec["tolerance"])


if __name__ == "__main__":
    unittest.main()
