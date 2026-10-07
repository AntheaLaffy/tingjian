#!/usr/bin/env python3
"""Compare independently produced little-endian FP32 tensors and emit metrics."""
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("actual")
parser.add_argument("reference")
parser.add_argument("--atol", type=float, default=0.01)
parser.add_argument("--relative-l2", type=float, default=0.002)
args = parser.parse_args()
a = np.fromfile(args.actual, dtype="<f4").astype(np.float64)
b = np.fromfile(args.reference, dtype="<f4").astype(np.float64)
assert a.shape == b.shape, (a.shape, b.shape)
assert len(a) and np.isfinite(a).all() and np.isfinite(b).all()
difference = a - b
result = {
    "elements": len(a),
    "max_abs_error": float(np.abs(difference).max()),
    "mean_abs_error": float(np.abs(difference).mean()),
    "relative_l2_error": float(np.linalg.norm(difference) / max(np.linalg.norm(b), 1e-12)),
    "cosine_similarity": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
}
result["passed"] = result["max_abs_error"] <= args.atol and result["relative_l2_error"] <= args.relative_l2
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)
