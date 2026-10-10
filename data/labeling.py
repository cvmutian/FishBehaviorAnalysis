import numpy as np

from features.feature_extractor import clip_statistics


def window_indicators(motion, geometry) -> dict:
    return clip_statistics(motion, geometry)


def prelabel(indicators: dict, rules: dict) -> int:
    speed = float(indicators["speed"])
    accel = float(indicators["accel"])
    yaw = _degrees(indicators["yaw"])
    bend = _degrees(indicators["bend"])
    tail = _degrees(indicators["tail_swing"])
    if yaw >= float(rules["yaw_turn_deg"]) and bend >= float(rules["bend_turn_deg"]):
        return 3
    if speed >= float(rules["speed_fast"]) and tail >= float(rules["tail_fast_deg"]):
        return 2
    if (
        speed < float(rules["speed_low"])
        and accel < float(rules["accel_low"])
        and yaw < float(rules["yaw_low_deg"])
        and bend < float(rules["bend_low_deg"])
        and tail < float(rules["tail_low_deg"])
    ):
        return 0
    if (
        float(rules["speed_low"]) <= speed < float(rules["speed_fast"])
        and accel < float(rules["accel_normal"])
        and yaw < float(rules["yaw_normal_deg"])
        and float(rules["bend_normal_min_deg"]) <= bend <= float(rules["bend_normal_max_deg"])
        and float(rules["tail_normal_min_deg"]) <= tail <= float(rules["tail_normal_max_deg"])
    ):
        return 1
    return -1


def resolve_label(indicators: dict, rules: dict, manual_label) -> tuple:
    if manual_label is not None:
        label = int(manual_label)
        if label < -1 or label > 3:
            raise ValueError(f"人工标签超出范围: {label}")
        if label < 0:
            return -1, "manual"
        return label, "manual"
    label = prelabel(indicators, rules)
    if label < 0:
        return -1, "review"
    return label, "prelabel"


def _degrees(value) -> float:
    return float(value) * 180.0 / np.pi
