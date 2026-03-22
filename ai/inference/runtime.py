# inference/run_analysis.py
import glob, csv, os
from pipeline import run_pipeline

def main():
    rows = [("video","t100","stroke","mode","spm","pace_100","dps_dpk","css")]
    for path in glob.glob("inference/*.mp4"):
        # Example defaults; customize per file if needed
        t100 = 61.2 if "bf" in path else 45.0
        stroke = "freestyle" if "bf" in path else "monofin"
        mode = "stroke" if stroke == "freestyle" else "kick"

        spm, out = run_pipeline(
            video_path=path,
            pool_len_m=50,
            t100_s=t100,
            stroke=stroke,
            mode=mode
        )
        rows.append((
            os.path.basename(path), t100, stroke, mode,
            round(spm,2),
            out.pace_100_str,
            round(out.dps_or_dpk_m or 0, 3),
            out.css_str or ""
        ))

    with open("inference/summary.csv","w",newline="") as f:
        csv.writer(f).writerows(rows)
    print("Saved: inference/summary.csv")

if __name__ == "__main__":
    main()
