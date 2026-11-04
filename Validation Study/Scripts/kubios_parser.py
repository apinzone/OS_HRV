import argparse
import csv
import re
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
 

TIME = {
    "Mean RR  (ms):": "mean_rr_ms",
    "Mean HR (beats/min):": "hr_bpm",
    "SDNN (ms):": "sdnn_ms",
    "RMSSD (ms):": "rmssd_ms",
    "NNxx (beats):": "nn50_count",
    "pNNxx (%):": "pnn50_percent",
}
NONLIN = {
    "SD1 (ms):": "sd1_ms",
    "SD2 (ms):": "sd2_ms",
    "SD2/SD1 ratio:": "_sd2_sd1",
    "Sample entropy (SampEn):": "sampen",
}
FREQ = {                 
    "Absolute powers": {"VLF (ms^2):": "vlf_power_ms2", "LF (ms^2):": "lf_power_ms2", "HF (ms^2):": "hf_power_ms2"},
    "Relative powers": {"LF (%):": "lf_percent", "HF (%):": "hf_percent"},
    "Normalized powers": {"LF (n.u.):": "lf_nu_percent", "HF (n.u.):": "hf_nu_percent"},
}
FREQ_TOP = {"Total power (ms^2):": "total_power_ms2", "LF/HF ratio:": "lf_hf_ratio"}
PEAKFREQ = {"VLF (Hz):": "vlf_peak_hz", "LF (Hz):": "lf_peak_hz", "HF (Hz):": "hf_peak_hz"}
EXTRA = {
    "PNS index:": "pns_index", "SNS index:": "sns_index", "Stress index:": "stress_index",
    "SD HR (beats/min):": "sd_hr_bpm", "Min HR (beats/min):": "min_hr_bpm", "Max HR (beats/min):": "max_hr_bpm",
    "RR tri index:": "rr_tri_index", "TINN (ms):": "tinn_ms",
    "Approximate entropy (ApEn):": "apen", "alpha 1:": "dfa_alpha1", "alpha 2:": "dfa_alpha2",
}
SETTINGS_KEYS = {
    "Detrending method:": "detrending", "Threshold for NNxx/pNNxx:": "nnxx_threshold",
    "VLF:": "vlf_band", "LF:": "lf_band", "HF:": "hf_band", "Interpolation rate:": "interp_rate",
    "Points in frequency-domain:": "freq_points", "Window width:": "window_width",
    "Window overlap:": "window_overlap", "AR model order:": "ar_order",
    "Apply detrending for nonlinear analysis:": "nonlin_detrend", "Entropy, embedding dimension:": "sampen_m",
    "Entropy, tolerance:": "sampen_r", "Beat correction:": "beat_correction",
    "Beats total:": "beats_total", "Beats corrected:": "beats_corrected",
    "Effective data length (s):": "effective_length_s",
}
 
 
def num(x):
    x = str(x).strip()
    try:
        return float(x)
    except ValueError:
        return np.nan
 
 
def parse_one(path):
    """Return dict(main=..., ar=..., extra=..., settings=...) for one Kubios CSV."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f)]
    text_lines = [r[0] if r else "" for r in rows]
 
    main, ar, extra, settings = {}, {}, {}, {}
    section, block = None, None
    for r in rows:
        if not r or not r[0]:
            continue
        lab = r[0]
        if lab.startswith("RR INTERVAL DATA"):                 # stop before the raw spectrum table
            break
        if lab.startswith("Time-Domain Results"):
            section = "time"; continue
        if lab.startswith("Frequency-Domain Results"):
            section = "freq"; block = None; continue
        if lab.startswith("Nonlinear Results"):
            section = "nonlin"; continue
        if lab.startswith("Results Overview"):
            section = "overview"; continue
        if lab in ("Absolute powers", "Relative powers", "Normalized powers", "Peak frequencies"):
            block = lab; continue
        vals = [num(v) for v in r[1:3]] + [np.nan, np.nan]
        v0, v1 = vals[0], vals[1]
 
        if lab in SETTINGS_KEYS:
            key = SETTINGS_KEYS[lab]
            if len(r) > 1 and r[1] not in ("",):
                settings.setdefault(key, r[1])
            continue
        if section == "time" and lab in TIME:
            main[TIME[lab]] = v0
        elif section == "time" and lab in EXTRA:
            extra[EXTRA[lab]] = v0
        elif section == "freq":
            if block == "Peak frequencies" and lab in PEAKFREQ:
                main[PEAKFREQ[lab]] = v0; ar[PEAKFREQ[lab]] = v1
            elif block in FREQ and lab in FREQ[block]:
                main[FREQ[block][lab]] = v0; ar[FREQ[block][lab]] = v1
            elif lab in FREQ_TOP:
                main[FREQ_TOP[lab]] = v0; ar[FREQ_TOP[lab]] = v1
        elif section == "nonlin" and lab in NONLIN:
            main[NONLIN[lab]] = v0
        elif lab in EXTRA:
            extra[EXTRA[lab]] = v0
 
    for line in text_lines:
        s = line.strip()
        for k, name in SETTINGS_KEYS.items():
            if s.startswith(k) and name not in settings:
                settings[name] = s[len(k):].strip()
    for line in text_lines[:8]:
        if line.startswith("Kubios"):
            settings["software"] = line
    ver = [l for l in text_lines[:8] if re.fullmatch(r"\d+\.\d+\.\d+", l.strip())]
    if ver:
        settings["version"] = ver[0]
 
    # Kubios reports SD2/SD1; ChronOS/NeuroKit2 columns are SD1/SD2
    sd21 = main.pop("_sd2_sd1", np.nan)
    main["sd1_sd2_ratio"] = 1.0 / sd21 if sd21 and not np.isnan(sd21) else np.nan
    return dict(main=main, ar=ar, extra=extra, settings=settings)
 

MAIN_ORDER = ["hr_bpm", "mean_rr_ms", "rmssd_ms", "pnn50_percent", "sdnn_ms", "sdsd_ms", "sd1_ms", "sd2_ms",
              "sd1_sd2_ratio", "sampen", "vlf_power_ms2", "lf_power_ms2", "hf_power_ms2", "total_power_ms2",
              "lf_hf_ratio"]

NAME = {"hr_bpm": "HR"}
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kubios-dir", default="Outputs", help="folder with the Kubios *_hrv.csv files (default: ./Outputs in the current directory)")
    ap.add_argument("--pattern", default="*_hrv.csv")
    ap.add_argument("--out", default="kubios_results.xlsx")
    ap.add_argument("--filename-template", default="synthetic_ecg_{n:03d}.edf",
                    help="how to build `filename` from the number in the Kubios file name")
    ap.add_argument("--use-stem", action="store_true",
                    help='filename = Kubios file name without "_rr_hrv.csv"/"_hrv.csv" + ".edf" '
                         '(for names without a single trailing number, e.g. fantasia_f1y01_seg1_rr_hrv.csv)')
    ap.add_argument("--chronos-metrics",
                    help="chronos_metrics.csv written by export_chronos_rr.py; adds a ChronOS sheet in the chronos_* layout")
    ap.add_argument("--prefix", default="kubios_",
                    help='column prefix; use "chronos_" to get literally the same headings as the ChronOS sheet')
    ap.add_argument("--merge", help="existing workbook (e.g. all_validity_data.xlsx); a COPY with the Kubios sheets added is written")
    ap.add_argument("--merged-out", default="all_validity_data_with_kubios.xlsx")
    args = ap.parse_args()
 
    files = sorted(Path(args.kubios_dir).glob(args.pattern))
    if not files:
        sys.exit(f"No files matching {args.pattern} in {args.kubios_dir}")
 
    main_rows, ar_rows, extra_rows, set_rows = [], [], [], []
    for p in files:
        if args.use_stem:
            fname = re.sub(r"(_rr)?_hrv\.csv$|\.csv$", "", p.name) + ".edf"
        else:
            m = re.search(r"(\d+)(?=_hrv\.csv$|\.csv$)", p.name)
            if not m:
                print(f"skipping (no number in name): {p.name}"); continue
            fname = args.filename_template.format(n=int(m.group(1)))
        try:
            d = parse_one(p)
        except Exception as e:                                  
            print(f"FAILED {p.name}: {e}"); continue
        beats = pd.to_numeric(d["settings"].get("beats_total"), errors="coerce")
        eff = pd.to_numeric(d["settings"].get("effective_length_s"), errors="coerce")
        d["main"]["sdsd_ms"] = d["main"].get("sd1_ms", np.nan) * np.sqrt(2)      # SD1 = SDSD/sqrt(2); Kubios has no SDSD row
        r = {"filename": fname,
             "duration_sec": eff,                       
             "sample_rate": np.nan,                   
             "total_peaks": beats + 1}                  
        r.update({f"{args.prefix}{NAME.get(k, k)}": d["main"].get(k, np.nan) for k in MAIN_ORDER})
        main_rows.append(r)
        ar_rows.append({"filename": fname, **{f"kubios_ar_{k}": d["ar"].get(k, np.nan)
                        for k in ["vlf_power_ms2", "lf_power_ms2", "hf_power_ms2", "total_power_ms2",
                                  "lf_hf_ratio", "lf_nu_percent", "hf_nu_percent"]}})
        extra_rows.append({"filename": fname, **{f"kubios_{k}": v for k, v in d["extra"].items()},
                           "kubios_nn50_count": d["main"].get("nn50_count", np.nan)})
        set_rows.append({"filename": fname, **d["settings"]})
 
    df = pd.DataFrame(main_rows).sort_values("filename").reset_index(drop=True)
    ar = pd.DataFrame(ar_rows).sort_values("filename").reset_index(drop=True)
    ex = pd.DataFrame(extra_rows).sort_values("filename").reset_index(drop=True)
    st = pd.DataFrame(set_rows).sort_values("filename").reset_index(drop=True)
 
    # QC: identical settings in every file, nothing missing
    cmp_cols = [c for c in st.columns if c not in ("filename", "beats_total", "beats_corrected", "effective_length_s")]
    st["settings_match_first"] = (st[cmp_cols].astype(str) == st[cmp_cols].astype(str).iloc[0]).all(axis=1)
    missing = df.drop(columns=["filename", "sample_rate"]).isna().sum()
    missing = missing[missing > 0]
 
    chron = None
    if args.chronos_metrics:
        cm = pd.read_csv(args.chronos_metrics)
        cm = cm[cm["status"].eq("success")] if "status" in cm else cm
        chron = pd.DataFrame({
            "filename": cm["filename"], "duration_sec": np.nan, "sample_rate": np.nan,
            "total_peaks": cm["num_peaks"], "chronos_HR": 60000.0 / cm["mean_rr_ms"],
            "chronos_mean_rr_ms": cm["mean_rr_ms"], "chronos_rmssd_ms": cm["rmssd_ms"],
            "chronos_pnn50_percent": cm["pNN50_%"], "chronos_sdnn_ms": cm["sdnn_ms"],
            "chronos_sdsd_ms": cm["sd1_ms"] * np.sqrt(2), "chronos_sd1_ms": cm["sd1_ms"],
            "chronos_sd2_ms": cm["sd2_ms"], "chronos_sd1_sd2_ratio": cm["sd1_sd2_ratio"],
            "chronos_sampen": cm["sample_entropy"], "chronos_vlf_power_ms2": cm["vlf_power_ms2"],
            "chronos_lf_power_ms2": cm["lf_power_ms2"], "chronos_hf_power_ms2": cm["hf_power_ms2"],
            "chronos_total_power_ms2": cm["total_power_ms2"], "chronos_lf_hf_ratio": cm["lf_hf_ratio"],
        }).sort_values("filename").reset_index(drop=True)
        chron.to_csv(Path(args.out).with_suffix("").as_posix() + "_chronos.csv", index=False)
 
    base = Path(args.out).with_suffix("")
    df.to_csv(f"{base}.csv", index=False)
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(args.out) as xw:
            if chron is not None:
                chron.to_excel(xw, sheet_name="ChronOS", index=False)
            df.to_excel(xw, sheet_name="Kubios", index=False)
            ar.to_excel(xw, sheet_name="Kubios_AR", index=False)
            ex.to_excel(xw, sheet_name="Extra", index=False)
            st.to_excel(xw, sheet_name="Settings", index=False)
    except ImportError:
        print("openpyxl not installed (pip install openpyxl): wrote CSV files only, no .xlsx")
        ar.to_csv(f"{base}_AR.csv", index=False); ex.to_csv(f"{base}_extra.csv", index=False)
        st.to_csv(f"{base}_settings.csv", index=False)
        if args.merge:
            sys.exit("--merge needs openpyxl (pip install openpyxl)")
 
    if args.merge:
        sheets = pd.read_excel(args.merge, sheet_name=None)
        with pd.ExcelWriter(args.merged_out) as xw:
            for name, sdf in sheets.items():
                sdf.to_excel(xw, sheet_name=name, index=False)
            df.to_excel(xw, sheet_name="Kubios", index=False)
            ar.to_excel(xw, sheet_name="Kubios_AR", index=False)
        print(f"Wrote {args.merged_out} (original {args.merge} untouched)")
 
    print(f"Parsed {len(df)} Kubios files -> {args.out}")
    print("Settings identical across all files:", bool(st["settings_match_first"].all()))
    if not st["settings_match_first"].all():
        print("  files with different settings:", ", ".join(st.loc[~st["settings_match_first"], "filename"]))
    print("Missing values per column:", dict(missing) if len(missing) else "none")
    if args.merge:
        base = pd.read_excel(args.merge, sheet_name=None)
        if "ChronOS" in base:
            miss = set(base["ChronOS"]["filename"]) - set(df["filename"])
            print(f"ChronOS files without a Kubios result: {len(miss)}" + (f" (e.g. {sorted(miss)[:3]})" if miss else ""))
 
 
if __name__ == "__main__":
    main()