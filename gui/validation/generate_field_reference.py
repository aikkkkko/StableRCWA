"""Generate a candidate corpus explicitly; ordinary tests only read the frozen one."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

from field_reference_cases import (ANCHOR_REVISION, COMPONENTS, CONTRACT_VERSION,
                                   FROZEN_DIR, NUMERIC_SOURCES, ROOT, case_specs, sample_case)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    arrays, records = {}, []
    for spec in case_specs():
        _, _, values, diagnostics = sample_case(spec)
        if diagnostics != dict(n_zero_modes=spec["expected_zero_modes"], n_schur=spec["expected_schur_layers"]):
            raise RuntimeError(f"{spec['id']}: expected zero/Schur path was not used")
        for key, value in values.items():
            if key not in ("phase_absolute", "phase_incident") and not np.isfinite(value).all():
                raise RuntimeError(f"{spec['id']}/{key}: nonfinite reference")
            arrays[spec["id"]+"__"+key] = value
        records.append(dict(parameters=spec, diagnostics=diagnostics,
                            arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype),
                                              order="C", key=spec["id"]+"__"+key)
                                    for key, value in values.items()}))
    archive = output / "reference_v1.npz"
    np.savez_compressed(archive, **arrays)
    # No wall-clock timestamp: identical code/environment reproduces the manifest.
    metadata = dict(contract_version=CONTRACT_VERSION, anchor_revision=ANCHOR_REVISION,
                    generator_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                    source_sha256={path: sha256(ROOT/path) for path in NUMERIC_SOURCES},
                    builder_sha256={path: sha256(ROOT/path) for path in (
                        "gui/validation/field_reference_cases.py", "gui/validation/generate_field_reference.py")},
                    archive=dict(filename=archive.name, sha256=sha256(archive), bytes=archive.stat().st_size),
                    components=list(COMPONENTS), complex_encoding="numpy complex128 (real + i imaginary)",
                    layout=dict(fields="[component,z,x]; x fastest", harmonics="[component,z,order]; order fastest",
                                phase="[component,z,x]; radians, NaN where phase_mask is true",
                                phase_mask="true = undefined/masked",
                                phase_comparison_mask="true = excluded from cross-backend phase comparison",
                                incident_amplitudes="[|E_inc|,|Z0H_inc|]",
                                wavenumbers="kx,ky,kzI,kzII include k0; inverse length units",
                                raw_fields="H=Z0H; preserve scalar/vector incident normalization"),
                    phase_options=dict(phase_floor=1e-6, unwrap=None),
                    comparison=dict(complex_fields="abs(actual-reference) <= atol+rtol*abs(reference)",
                                    structural_zero="Only abs(field)<=field atol; raw phase is not an oracle",
                                    phase_exclusion="phase_mask OR abs(field)<=10*(atol+rtol*abs(field)) OR structural zero component",
                                    phase_metric="abs(exp(i*actual_phase)-exp(i*reference_phase))",
                                    phase_tolerance="2*(atol+rtol*abs(reference_field))/abs(reference_field)+5e-14"),
                    cases=records)
    (output / "reference_v1.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Candidate directory; frozen data requires --refresh-baseline")
    parser.add_argument("--refresh-baseline", action="store_true", help="Explicitly allow updating the checked-in baseline")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == FROZEN_DIR.resolve() and not args.refresh_baseline:
        parser.error("Frozen corpus is read-only by default; use a candidate directory or explicitly --refresh-baseline")
    if output != FROZEN_DIR.resolve() and args.refresh_baseline:
        parser.error("--refresh-baseline applies only to gui/validation/field_reference")
    manifest = generate(output)
    print(f"{len(manifest['cases'])} cases, {manifest['archive']['bytes']} bytes: {output}")


if __name__ == "__main__":
    main()
