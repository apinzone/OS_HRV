import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def parse_list(s):
    return np.array(ast.literal_eval(re.sub(r"np\.float\d*\(([^)]*)\)", r"\1", str(s))), float)


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
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--truth", default=str(HERE / "fantasia_segments" / "fantasia_segments_truth.csv"))
    ap.add_argument("--out-prefix", default="scores")
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    args = ap.parse_args()

    truth = pd.read_csv(args.truth)
    truth["filename"] = truth["segment"] + ".edf"
    for sheet, tag in [("ChronOS_Results", "ChronOS"), ("NeuroKit2_Results", "NeuroKit2")]:
        df = pd.read_excel(args.workbook, sheet_name=sheet)
        rows = []
        for _, r in df.iterrows():
            t = truth[truth.filename == r.filename].beat_time_ms.values
            pk = parse_list(r.r_peak_timestamps_sec) * 1000.0
            tp, fp, fn = match(pk, t, args.tolerance_ms)
            rows.append({"segment": r.filename, "true_beats": len(t), "detected": len(pk), "TP": tp, "FP": fp, "FN": fn,
                         "sensitivity": tp / (tp + fn), "ppv": tp / (tp + fp)})
        d = pd.DataFrame(rows)
        d.to_csv(f"{args.out_prefix}_{tag}.csv", index=False)
        T = d[["true_beats", "TP", "FP", "FN"]].sum()
        s, p = T.TP / (T.TP + T.FN), T.TP / (T.TP + T.FP)
        print(f"{tag:10s} TP {int(T.TP)}  FP {int(T.FP)}  FN {int(T.FN)}  sensitivity {s:.4%}  PPV {p:.4%}  F1 {2*s*p/(s+p):.4f}")


if __name__ == "__main__":
    main()