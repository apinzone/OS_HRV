import os
import numpy as np
import neurokit2 as nk
import pyedflib
from pyedflib import highlevel

# -------- Settings --------
OUTDIR = "synthetic_edf"
NAME = "ECG_clean_3min"
fs = 256                # Hz
duration_s = 180        # seconds
hr_bpm = 60
noise = 0.0
lf, hf = 0.10, 0.25
lfampl, hfampl = 0.5, 1.0
seed = 42

os.makedirs(OUTDIR, exist_ok=True)
edf_path = os.path.join(OUTDIR, f"{NAME}.edf")

# -------- Target length --------
n_target = int(round(duration_s * fs))
n_target = max(n_target, fs)

# -------- Chunked simulate + crossfade stitch --------
CHUNK = 4096                     # buggy NK2 cap
XFADE = min(128, CHUNK // 8)    # crossfade length (samples)
rng = np.random.default_rng(seed)

def simulate_chunk(n=CHUNK):
    # per-chunk seed so chunks are reproducible but not identical
    chunk_seed = int(rng.integers(0, 2**31 - 1))
    x = nk.ecg_simulate(
        length=int(n),
        sampling_rate=int(fs),
        heart_rate=float(hr_bpm),
        method="ecgsyn",
        noise=float(noise),
        lf=float(lf), hf=float(hf), lfampl=float(lfampl), hfampl=float(hfampl),
        random_state=chunk_seed
    )
    return np.asarray(x, dtype=float).ravel()

def stitch_with_crossfade(parts, xfade=XFADE):
    if len(parts) == 1:
        return parts[0]
    out = parts[0].copy()
    for p in parts[1:]:
        a = out[-xfade:]
        b = p[:xfade]
        w = np.linspace(0.0, 1.0, xfade)  # fade from a -> b
        seam = (1 - w) * a + w * b
        out = np.concatenate([out[:-xfade], seam, p[xfade:]])
    return out

# build enough chunks then trim to exact length
pieces = []
remaining = n_target
while remaining > 0:
    c = simulate_chunk(CHUNK)
    take = min(remaining, c.size)
    pieces.append(c[:take] if take < c.size else c)
    remaining -= take

ecg = stitch_with_crossfade(pieces, XFADE)
ecg = ecg[:n_target]  # exact length
ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
print(f"[simulate] final length={ecg.size} (expected {n_target}), duration={ecg.size/fs:.3f}s")

# -------- EDF+ write (high-level), tidy phys limits to avoid 8-char warning --------
# Use neat limits (±5 mV) so headers stay within EDF’s 8-char precision
phys_min, phys_max = -5.0, 5.0

sig_header = highlevel.make_signal_header(
    label="ECG",
    dimension="mV",
    sample_frequency=float(fs),
    physical_min=phys_min,
    physical_max=phys_max,
    digital_min=-32768,
    digital_max=32767
)
file_header = highlevel.make_header(
    patientname="Synthetic",
    recording_additional="NeuroKit2 ECGSYN (chunked+crossfade)"
)

highlevel.write_edf(
    edf_path,
    signals=[ecg.astype(float)],
    signal_headers=[sig_header],
    header=file_header,
    digital=False
)
print(f"[write] Saved: {edf_path}")

# -------- Sanity check --------
with pyedflib.EdfReader(edf_path) as f:
    fs_i = f.getSampleFrequency(0)
    n_i = f.getNSamples()[0]
    print(f"[check] fs={fs_i} Hz, n_samples={n_i}, duration={n_i/fs_i:.2f} s")
