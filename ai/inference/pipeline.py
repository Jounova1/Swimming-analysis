# ai/inference/pipeline.py
from typing import Optional, Tuple
from DPS_calc import DPSCalculator, SwimInputs
from stroke_rate_analysis import estimate_spm

def run_pipeline(
    video_path: str,
    *,
    pool_len_m: float,
    t100_s: float,
    t200_s: Optional[float] = None,
    t400_s: Optional[float] = None,
    stroke: str = "auto",               # freestyle/backstroke/breaststroke/butterfly/monofin/underwater/auto
    mode: Optional[str] = None,         # "stroke" or "kick" or None(auto from stroke)
    roi: Optional[Tuple[int,int,int,int]] = None,
    min_hz: float = 0.4,
    max_hz: float = 2.0,
    display: bool = False
):
    """
    Returns: (spm, SwimOutputs)
    """
    # 1) Decide rate mode
    if mode is None or mode == "auto":
        if stroke in ("monofin", "underwater"):
            rate_mode = "kick"
        else:
            rate_mode = "stroke"
    else:
        rate_mode = mode

    # 2) Estimate SPM from video
    spm = estimate_spm(
        video_path=video_path,
        mode=rate_mode,
        roi=roi,
        min_hz=min_hz,
        max_hz=max_hz,
        display=display
    )

    # 3) Build inputs for DPS calculator
    stroke_type = None if stroke == "auto" else stroke
    stroke_rate_spm = spm if rate_mode == "stroke" else None
    kick_rate_spm = spm if rate_mode == "kick" else None
    is_monofin = (stroke == "monofin")
    is_underwater = (stroke == "underwater")

    calc = DPSCalculator()
    out = calc.compute(SwimInputs(
        pool_len_m=pool_len_m,
        t100_s=t100_s,
        t200_s=t200_s,
        t400_s=t400_s,
        stroke_rate_spm=stroke_rate_spm,
        kick_rate_spm=kick_rate_spm,
        stroke_type=stroke_type,
        is_monofin=is_monofin,
        is_underwater=is_underwater
    ))
    return spm, out