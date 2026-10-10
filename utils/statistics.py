def summarize_school_status(results, behavior_names) -> dict:
    counts = {name: 0 for name in behavior_names}
    num_insufficient = 0
    confidences = []
    kinematic_keys = ("speed", "accel", "yaw", "bend", "tail_swing")
    kinematic_values = {key: [] for key in kinematic_keys}
    for item in results:
        name = item.get("behavior", item.get("behavior_name"))
        behavior_id = item.get("behavior_id", None)
        if name not in counts or (behavior_id is not None and int(behavior_id) < 0):
            num_insufficient += 1
            continue
        counts[name] += 1
        if item.get("confidence") is not None:
            confidences.append(float(item["confidence"]))
        kinematics = item.get("kinematics") or {}
        for key in kinematic_keys:
            if key in kinematics:
                kinematic_values[key].append(float(kinematics[key]))

    num_valid = sum(counts.values())
    behaviors = {}
    for name in behavior_names:
        count = counts[name]
        ratio = (count / num_valid) if num_valid > 0 else 0.0
        behaviors[name] = {"count": count, "ratio": ratio}

    if num_valid == 0:
        dominant_behavior = None
        dominant_ratio = 0.0
    else:
        dominant_behavior = max(behavior_names, key=lambda name: counts[name])
        dominant_ratio = counts[dominant_behavior] / num_valid

    activity = {
        "dominant_behavior": dominant_behavior,
        "dominant_ratio": dominant_ratio,
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
    }
    for key, values in kinematic_values.items():
        activity[key] = (sum(values) / len(values)) if values else None
    return {
        "num_total": len(results),
        "num_valid": num_valid,
        "num_insufficient": num_insufficient,
        "behaviors": behaviors,
        "activity": activity,
    }
