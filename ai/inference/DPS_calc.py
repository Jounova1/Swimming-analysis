# ai/inference/DPS_calc.py
from dataclasses import dataclass
from typing import Optional, Dict, Literal

StrokeType = Literal[
    "freestyle", "backstroke", "breaststroke", "butterfly",
    "monofin", "underwater", "unknown"
]

def pace_to_str(seconds_per_100: float) -> str:
    m = int(seconds_per_100 // 60)
    s = seconds_per_100 % 60
    return f"{m}:{s:04.1f}/100m"

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

@dataclass
class SwimInputs:
    pool_len_m: float
    t100_s: float
    t50_s: Optional[float] = None
    t200_s: Optional[float] = None
    t400_s: Optional[float] = None
    stroke_rate_spm: Optional[float] = None
    kick_rate_spm: Optional[float] = None
    stroke_type: Optional[StrokeType] = None  # explicit or "auto"
    is_monofin: Optional[bool] = None
    is_underwater: Optional[bool] = None

@dataclass
class SwimOutputs:
    stroke_type: StrokeType
    speed_mps: float
    speed_kmh: float
    pace_100_s: float
    pace_100_str: str
    dps_or_dpk_m: Optional[float]
    css_s_per_100: Optional[float]
    css_mps: Optional[float]
    css_str: Optional[str]
    zones_s_per_100: Optional[Dict[str, float]]

class DPSCalculator:
    """Distance Per Stroke/Kick calculator + CSS/zones."""

    # ---- core math ----
    def speed_from_t100(self, t100_s: float) -> float:
        if t100_s <= 0:
            raise ValueError("t100_s must be > 0")
        return 100.0 / t100_s

    def pace_from_speed(self, speed_mps: float) -> float:
        if speed_mps <= 0:
            raise ValueError("speed must be > 0")
        return 100.0 / speed_mps

    def css_from_200_400(self, t200_s: float, t400_s: float) -> float:
        if t400_s <= t200_s:
            raise ValueError("t400_s must be greater than t200_s")
        return (t400_s - t200_s) / 2.0

    def distance_per_cycle(self, speed_mps: float, rate_spm: float) -> float:
        if rate_spm is None or rate_spm <= 0:
            raise ValueError("rate_spm must be > 0")
        cps = rate_spm / 60.0
        return speed_mps / cps

    def make_zones(self, css_s_per_100: float) -> Dict[str, float]:
        return {
            "A1 (Easy)":       css_s_per_100 + 8.0,
            "A2 (Endurance)":  css_s_per_100 + 4.0,
            "Threshold (T)":   css_s_per_100 + 0.0,
            "VO2":             clamp(css_s_per_100 - 4.0, 10.0, 999.0),
            "Sprint":          clamp(css_s_per_100 - 8.0, 8.0, 999.0),
        }

    # ---- heuristic stroke detection ----
    def detect_stroke(
        self,
        t100_s: float,
        pool_len_m: float,
        stroke_rate_spm: Optional[float],
        kick_rate_spm: Optional[float],
        hint_monofin: Optional[bool],
        hint_underwater: Optional[bool]
    ) -> StrokeType:
        if hint_underwater:
            return "underwater"
        if hint_monofin:
            return "monofin"

        speed_mps = self.speed_from_t100(t100_s)

        if kick_rate_spm and (stroke_rate_spm is None or kick_rate_spm > 0):
            if t100_s < 55 and kick_rate_spm >= 50:
                return "underwater"
            return "monofin"

        if stroke_rate_spm and stroke_rate_spm > 0:
            try:
                dps_est = self.distance_per_cycle(speed_mps, stroke_rate_spm)
            except Exception:
                dps_est = None
            if dps_est is not None:
                if dps_est < 1.6 and 20 <= stroke_rate_spm <= 60:
                    return "breaststroke"
                if 1.5 <= dps_est <= 2.1 and 25 <= stroke_rate_spm <= 60:
                    return "butterfly"
                if 1.6 <= dps_est <= 2.1 and 35 <= stroke_rate_spm <= 80:
                    return "backstroke"
                if dps_est >= 1.6 and stroke_rate_spm >= 40:
                    return "freestyle"

        return "unknown"

    # ---- main compute ----
    def compute(self, inp: SwimInputs) -> SwimOutputs:
        speed_mps = self.speed_from_t100(inp.t100_s)
        pace_100_s = self.pace_from_speed(speed_mps)
        speed_kmh = speed_mps * 3.6

        chosen = inp.stroke_type or "unknown"
        if chosen in (None, "unknown"):
            chosen = self.detect_stroke(
                inp.t100_s, inp.pool_len_m,
                inp.stroke_rate_spm, inp.kick_rate_spm,
                inp.is_monofin, inp.is_underwater
            )

        dps_or_dpk = None
        if chosen in ("monofin", "underwater"):
            if inp.kick_rate_spm and inp.kick_rate_spm > 0:
                dps_or_dpk = self.distance_per_cycle(speed_mps, inp.kick_rate_spm)
        else:
            if inp.stroke_rate_spm and inp.stroke_rate_spm > 0:
                dps_or_dpk = self.distance_per_cycle(speed_mps, inp.stroke_rate_spm)

        css_s_per_100 = css_mps = None
        css_str = None
        zones = None
        if inp.t200_s and inp.t400_s:
            css_s_per_100 = self.css_from_200_400(inp.t200_s, inp.t400_s)
            css_mps = 100.0 / css_s_per_100
            css_str = pace_to_str(css_s_per_100)
            zones = self.make_zones(css_s_per_100)

        return SwimOutputs(
            stroke_type=chosen,
            speed_mps=speed_mps,
            speed_kmh=speed_kmh,
            pace_100_s=pace_100_s,
            pace_100_str=pace_to_str(pace_100_s),
            dps_or_dpk_m=dps_or_dpk,
            css_s_per_100=css_s_per_100,
            css_mps=css_mps,
            css_str=css_str,
            zones_s_per_100=zones
        )