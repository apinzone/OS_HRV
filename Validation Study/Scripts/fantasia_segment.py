"""
fantasia_segments.py

Cut fixed 5-minute ECG segments out of the Fantasia young-subject records (f1y01-f1y05) and write
(1) one single-channel EDF per segment (input for batch_processor / export_chronos_rr.py) and
(2) the expert beat annotations that fall inside each segment (ground truth, times relative to segment start).

The raw records carry a ~8 mV DC offset. ChronOS' height threshold is absolute (0.55 x (max - median)), so on an
uncentred signal it lies below the baseline and T waves are detected as well. Each segment is therefore shifted by
its median (constant offset removal, no filtering) unless --no-center is given.

Segments are chosen by rule, not by inspection of the data: minutes 30-35 and 60-65 of every record.

Usage (run from "Validation Study/"):
    python fantasia_segments.py
    python fantasia_segments.py --records f1y01 f1y02 --windows 30-35 60-65 --out-dir fantasia_segments

Outputs (--out-dir):
    fantasia_<record>_seg<k>.edf
    fantasia_segments_truth.csv      segment, beat_time_ms (relative to segment start), beat_type, sample_in_segment
    fantasia_segments_summary.csv    one row per segment: n annotated beats, n non-'N' beats, mean HR
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
import wfdb

HERE = Path(__file__).resolve().parent
NON_BEAT = {"+", "~", "|", "x", "(", ")", "p", "t", "u", "`", "'", "^", "!", "[", "]", "{", "}", "@"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE / "fantasia_validation"))
    ap.add_argument("--out-dir", default=str(HERE / "fantasia_segments"))
    ap.add_argument("--records", nargs="+", default=[f"f1y0{k}" for k in range(1, 6)])
    ap.add_argument("--no-center", action="store_true",
                    help="keep the raw DC offset (~8 mV in these records); by default each segment is shifted by its median")
    ap.add_argument("--windows", nargs="+", default=["30-35", "60-65"], help="minute windows, e.g. 30-35")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wins = [tuple(float(x) for x in w.split("-")) for w in args.windows]

    truth_rows, summary_rows = [], []
    for rec in args.records:
        hdr = wfdb.rdheader(str(data_dir / rec))
        fs = int(hdr.fs)
        ecg_idx = hdr.sig_name.index("ECG")                       # RESP is channel 0 in these records
        ann = wfdb.rdann(str(data_dir / rec), "ecg")
        for k, (m0, m1) in enumerate(wins, 1):
            a, b = int(m0 * 60 * fs), int(m1 * 60 * fs)
            sig = wfdb.rdrecord(str(data_dir / rec), sampfrom=a, sampto=b, channels=[ecg_idx]).p_signal[:, 0]
            if not args.no_center:
                sig = sig - np.median(sig)             # constant shift only; no filtering
            name = f"fantasia_{rec}_seg{k}"
            sh = pyedflib.highlevel.make_signal_header(
                "ECG", dimension="mV", sample_frequency=fs,
                physical_min=float(np.floor(sig.min() - 0.1)), physical_max=float(np.ceil(sig.max() + 0.1)),
                digital_min=-32768, digital_max=32767)
            pyedflib.highlevel.write_edf(str(out_dir / f"{name}.edf"), [sig], [sh], header={"patientname": rec})

            keep = (ann.sample >= a) & (ann.sample < b) & ~np.isin(ann.symbol, list(NON_BEAT))
            samp = ann.sample[keep] - a
            sym = np.array(ann.symbol)[keep]
            for s_, y in zip(samp, sym):
                truth_rows.append({"segment": name, "beat_time_ms": s_ / fs * 1000.0, "beat_type": y,
                                   "sample_in_segment": int(s_)})
            rr = np.diff(samp) / fs * 1000.0
            summary_rows.append({"segment": name, "record": rec, "window_min": f"{m0:g}-{m1:g}",
                                 "samples": len(sig), "annotated_beats": len(samp),
                                 "non_N_beats": int((sym != "N").sum()),
                                 "mean_hr_bpm": 60000.0 / rr.mean() if len(rr) else np.nan})
    pd.DataFrame(truth_rows).to_csv(out_dir / "fantasia_segments_truth.csv", index=False)
    s = pd.DataFrame(summary_rows)
    s.to_csv(out_dir / "fantasia_segments_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()