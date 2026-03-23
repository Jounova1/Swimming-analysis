# ai/inference/run_analysis.py
import glob, csv, os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pipeline import run_pipeline

def main():
    rows = [("video","t100","stroke","mode","spm","pace_100","dps_dpk","css")]
    for path in glob.glob(os.path.join(SCRIPT_DIR, "*.mp4")):
        fname = os.path.basename(path).lower()
        # Simple demo defaults; customize per file if needed
        if "underwater" in fname or "mono" in fname:
            t100 = 45.0; stroke = "monofin"; mode = "kick"; min_hz, max_hz = 0.5, 2.2
        else:
            t100 = 61.2; stroke = "freestyle"; mode = "stroke"; min_hz, max_hz = 0.4, 1.6

        spm, out = run_pipeline(
            video_path=path,
            pool_len_m=50,
            t100_s=t100,
            stroke=stroke,
            mode=mode,
            min_hz=min_hz,
            max_hz=max_hz,
            display=False
        )
        rows.append((
            os.path.basename(path), t100, stroke, mode,
            round(spm,2),
            out.pace_100_str,
            round(out.dps_or_dpk_m or 0, 3),
            out.css_str or ""
        ))

    csv_path = os.path.join(SCRIPT_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()