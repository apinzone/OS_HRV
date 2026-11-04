import argparse
import contextlib
import io
import os
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
 
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # batch_processor.py
sys.path.insert(0, str(HERE.parent))          # analyzer.py, functions.py (repo root)
 
import batch_processor as bp                  # noqa: E402  (needs neurokit2 installed, as for the batch run)
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edf-dir", default=str(HERE / "validation_synthetic_ecg"))
    ap.add_argument("--out-dir", default=str(HERE / "chronos_rr_export"))
    ap.add_argument("--height-fraction", type=float, default=0.55,
                    help="ChronOS peak-height fraction of (max - median); 0.55 = validation-study default")
    ap.add_argument("--check", action="store_true",
                    help="compare re-run metrics with validation_results.xlsx (ChronOS_Results sheet)")
    args = ap.parse_args()
 
    if args.height_fraction != 0.55:
        _orig = bp.calculate_adaptive_ecg_params
 
        def _patched(sig, fs):
            d = _orig(sig, fs)
            d["ecg_height"] = args.height_fraction * (np.max(sig) - np.median(sig))
            d["ecg_prominence"] = 0.6 * d["ecg_height"]
            return d
        bp.calculate_adaptive_ecg_params = _patched
        print(f"Peak-height fraction set to {args.height_fraction} (default 0.55); prominence stays 0.6 x height")
 
    edf_dir, out_dir = Path(args.edf_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edf_files = sorted(edf_dir.glob("*.edf"))
    if not edf_files:
        sys.exit(f"No .edf files found in {edf_dir}")
 
    rows = []
    for i, edf in enumerate(edf_files, 1):
        with contextlib.redirect_stdout(io.StringIO()):        
            res = bp.analyze_with_chronos(str(edf), edf.name)
        if res is None:
            print(f"[{i}/{len(edf_files)}] FAILED: {edf.name}")
            continue
        np.savetxt(out_dir / f"{edf.stem}_rr.txt",
                   np.asarray(res["rr_intervals_ms"], dtype=float), fmt="%.4f")
        np.savetxt(out_dir / f"{edf.stem}_peaktimes.txt",
                   np.asarray(res["r_peak_timestamps_sec"], dtype=float), fmt="%.6f")
        rows.append({k: v for k, v in res.items()
                     if k not in ("rr_intervals_ms", "r_peak_timestamps_sec")})
 
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "chronos_metrics.csv", index=False)
    print(f"Exported RR for {len(df)}/{len(edf_files)} files to {out_dir}")
 
    if args.check:
        xlsx = HERE / "validation_results.xlsx"
        old = pd.read_excel(xlsx, "ChronOS_Results")
        old["filename"] = old["filename"].map(lambda f: os.path.basename(str(f)))
        merged = df.merge(old, on="filename", suffixes=("_new", "_old"))
        shared = [c for c in old.columns
                  if c not in ("filename", "status", "r_peak_timestamps_sec", "rr_intervals_ms")
                  and f"{c}_new" in merged.columns]
        print(f"\nRe-run vs {xlsx.name}: {len(merged)} files matched by name")
        print("max |re-run - saved| per metric (0 = identical):")
        for c in shared:
            diff = np.nanmax(np.abs(merged[f"{c}_new"].astype(float) - merged[f"{c}_old"].astype(float)))
            print(f"  {c:<18s} {diff:.3g}")
 
 
if __name__ == "__main__":
    main()
 