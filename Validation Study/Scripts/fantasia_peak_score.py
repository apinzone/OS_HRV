
# Score ChronOS R-peak detections on the Fantasia segments against the expert annotations
# (+-50 ms tolerance, same logic as calculate_peak_metrics in HRV_Validity.R).

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def match(det_ms, true_ms, tol=50.0):
    det, tru = np.sort(det_ms), np.sort(true_ms)
    used = np.zeros(len(tru), bool)
    tp = fp = 0
    for d in det:
        diff = np.abs(tru - d)
        ok = np.where((diff <= tol) & ~used)[0]
        if len(ok):
            used[ok[np.argmin(diff[ok])]] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, int((~used).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-dir", default=str(HERE / "fantasia_segments"))
    ap.add_argument("--export-dir", default=None, help="default: <seg-dir>/chronos_export")
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    args = ap.parse_args()
    seg = Path(args.seg_dir)
    exp = Path(args.export_dir) if args.export_dir else seg / "chronos_export"
    truth = pd.read_csv(seg / "fantasia_segments_truth.csv")

    rows = []
    for name, t in truth.groupby("segment"):
        pk = np.loadtxt(exp / f"{name}_peaktimes.txt", ndmin=1) * 1000.0
        tp, fp, fn = match(pk, t.beat_time_ms.values, args.tolerance_ms)
        sens, ppv = tp / (tp + fn), tp / (tp + fp)
        rows.append({"segment": name, "true_beats": len(t), "detected": len(pk), "TP": tp, "FP": fp, "FN": fn,
                     "sensitivity": sens, "ppv": ppv, "f1": 2 * sens * ppv / (sens + ppv),
                     "non_N_beats": int((t.beat_type != "N").sum())})
    d = pd.DataFrame(rows)
    d.to_csv(seg / "fantasia_peak_detection.csv", index=False)
    print(d.round(4).to_string(index=False))
    T = d[["true_beats", "TP", "FP", "FN"]].sum()
    s, p = T.TP / (T.TP + T.FN), T.TP / (T.TP + T.FP)
    print(f"\nPooled: {int(T.true_beats)} beats, TP {int(T.TP)}, FP {int(T.FP)}, FN {int(T.FN)}; "
          f"sensitivity {s:.4f}, PPV {p:.4f}, F1 {2*s*p/(s+p):.4f}")


if __name__ == "__main__":
    main()