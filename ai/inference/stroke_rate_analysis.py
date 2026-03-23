# ai/inference/stroke_rate_analysis.py
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Optional, Tuple

def _bandpass(sig, fs, low, high, order=4):
    nyq = 0.5 * fs
    low /= nyq; high /= nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)

def _autocorr_period(x, fs, min_hz, max_hz):
    x = (x - np.mean(x))
    x = x / (np.std(x) + 1e-9)
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size // 2:]
    min_lag = max(int(fs / max_hz), 1)
    max_lag = min(int(fs / min_hz), len(ac) - 1)
    if max_lag <= min_lag:
        return None
    roi = ac[min_lag:max_lag]
    peak_idx = int(np.argmax(roi)) + min_lag
    return peak_idx / fs

def estimate_spm(
    video_path: str,
    mode: str = "stroke",                  # "stroke" or "kick"
    roi: Optional[Tuple[int,int,int,int]] = None,  # (x,y,w,h)
    sample_ratio: int = 1,
    min_hz: float = 0.4,
    max_hz: float = 2.0,
    display: bool = False
) -> float:
    """
    Returns cycles per minute (SPM: strokes/min OR kicks/min).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fs = fps / max(1, sample_ratio)

    ok, prev = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read first frame")

    H, W = prev.shape[:2]
    if roi:
        x, y, w, h = roi
        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
    else:
        w, h = W // 2, H // 2
        x, y = (W - w)//2, (H - h)//2

    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_g = cv2.GaussianBlur(prev_g, (5,5), 0)

    energies = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (i % max(1, sample_ratio)) != 0:
            i += 1
            continue

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5,5), 0)
        diff = cv2.absdiff(g[y:y+h, x:x+w], prev_g[y:y+h, x:x+w])
        energies.append(float(np.mean(diff)))

        if display:
            vis = frame.copy()
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(vis, f"Energy:{energies[-1]:.2f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.imshow("video", vis); cv2.imshow("diff", diff)
            if cv2.waitKey(1) & 0xFF == 27: break

        prev_g = g
        i += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()

    energies = np.asarray(energies, dtype=np.float32)
    if len(energies) < fs * 5:
        print("[Warning] Very short clip; use 5–10 s for stability.")
    if len(energies) < 10:
        raise RuntimeError("Not enough frames to estimate rate.")

    try:
        sig = _bandpass(energies - np.mean(energies), fs, min_hz, max_hz)
    except ValueError:
        sig = energies - np.mean(energies)

    period = _autocorr_period(sig, fs, min_hz, max_hz)
    if not period or period <= 0:
        raise RuntimeError("Could not find a stable periodic rate. Adjust ROI / min-max Hz.")
    hz = 1.0 / period
    spm = hz * 60.0
    return float(spm)
