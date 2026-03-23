# ai/inference/main.py
import os
import sys
import argparse

# Ensure local imports work when running from this folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Resilient import for optional ROI detector (falls back to None)
try:
    from detect_swimmer import get_initial_roi  # should return Optional[(x,y,w,h)]
except Exception:
    def get_initial_roi(video_path: str):
        return None

from pipeline import run_pipeline

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video (e.g., bf.mp4)")
    ap.add_argument("--pool_len", type=float, default=50.0)
    ap.add_argument("--t100", type=float, required=True, help="100 m time in seconds")
    ap.add_argument("--t200", type=float, help="200 m time (optional; with 400 for CSS)")
    ap.add_argument("--t400", type=float, help="400 m time (optional; with 200 for CSS)")
    ap.add_argument("--stroke", default="auto",
                    choices=["freestyle","backstroke","breaststroke","butterfly","monofin","underwater","auto"])
    ap.add_argument("--mode", default="auto", choices=["auto","stroke","kick"])
    ap.add_argument("--roi", nargs=4, type=int, metavar=("X","Y","W","H"), default=None)
    ap.add_argument("--min_hz", type=float, default=0.4)
    ap.add_argument("--max_hz", type=float, default=2.0)
    ap.add_argument("--display", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Uncomment for quick sanity check
    # print("[DEBUG args]:", vars(args))

    # ROI priority: CLI > detector > None
    cli_roi = tuple(args.roi) if args.roi else None
    det_roi = get_initial_roi(args.video)
    roi = cli_roi if cli_roi is not None else det_roi

    spm, out = run_pipeline(
        video_path=args.video,
        pool_len_m=args.pool_len,
        t100_s=args.t100,
        t200_s=args.t200,
        t400_s=args.t400,
        stroke=args.stroke,
        mode=args.mode,
        roi=roi,
        min_hz=args.min_hz,
        max_hz=args.max_hz,
        display=args.display
    )

    label = "Kick rate (kicks/min)" if (args.mode == "kick" or args.stroke in ("monofin","underwater")) \
            else "Stroke rate (strokes/min)"
    print("\n=== Video → Rate → DPS Results ===")
    print(f"{label}: {spm:.2f}")
    print(f"Detected/Chosen stroke: {out.stroke_type}")
    print(f"Speed: {out.speed_mps:.3f} m/s  ({out.speed_kmh:.2f} km/h)")
    print(f"Pace:  {out.pace_100_str}")

    if out.dps_or_dpk_m is not None:
        unit = "m/kick" if out.stroke_type in ("monofin","underwater") else "m/stroke"
        print(f"DPS/DPK: {out.dps_or_dpk_m:.2f} {unit}")
    else:
        print("(Provide the relevant rate to compute DPS/DPK)")

    if out.css_s_per_100:
        print(f"\nCSS: {out.css_str}  (~{out.css_mps:.3f} m/s)")
        print("Training zones (pace per 100 m):")
        for name, sec100 in out.zones_s_per_100.items():
            m = int(sec100 // 60); s = sec100 % 60
            print(f"- {name}: {m}:{s:04.1f}/100m")
    else:
        print("\n(To enable CSS/zones, pass --t200 and --t400).") 