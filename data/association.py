import numpy as np


def linear_sum_assignment(cost: np.ndarray):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T.copy()
    else:
        cost = cost.copy()
    n_rows, n_cols = cost.shape
    u = np.zeros(n_rows + 1)
    v = np.zeros(n_cols + 1)
    pointer = np.zeros(n_cols + 1, dtype=int)
    way = np.zeros(n_cols + 1, dtype=int)
    for row in range(1, n_rows + 1):
        pointer[0] = row
        min_value = np.full(n_cols + 1, np.inf)
        used = np.zeros(n_cols + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = pointer[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, n_cols + 1):
                if used[candidate]:
                    continue
                current = cost[current_row - 1, candidate - 1] - u[current_row] - v[candidate]
                if current < min_value[candidate]:
                    min_value[candidate] = current
                    way[candidate] = column
                if min_value[candidate] < delta:
                    delta = min_value[candidate]
                    next_column = candidate
            for candidate in range(n_cols + 1):
                if used[candidate]:
                    u[pointer[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_value[candidate] -= delta
            column = next_column
            if pointer[column] == 0:
                break
        while True:
            previous = way[column]
            pointer[column] = pointer[previous]
            column = previous
            if column == 0:
                break
    rows = []
    cols = []
    for column in range(1, n_cols + 1):
        if pointer[column] != 0:
            rows.append(pointer[column] - 1)
            cols.append(column - 1)
    row_index = np.asarray(rows, dtype=int)
    col_index = np.asarray(cols, dtype=int)
    if transposed:
        return col_index, row_index
    return row_index, col_index


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    intersection = width * height
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def keypoints_to_box(keypoints: np.ndarray) -> np.ndarray:
    xy = np.asarray(keypoints)[:, :2]
    finite = np.isfinite(xy).all(axis=-1)
    if not np.any(finite):
        return np.zeros(4, dtype=np.float64)
    points = xy[finite]
    return np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()], dtype=np.float64)


def association_cost(track_keypoints, detection_keypoints, detection_box, config: dict) -> float:
    previous = np.asarray(track_keypoints, dtype=np.float64)
    current = np.asarray(detection_keypoints, dtype=np.float64)
    previous_xy = previous[:, :2]
    current_xy = current[:, :2]
    weights = config["association"]
    center_distance = np.linalg.norm(previous_xy.mean(axis=0) - current_xy.mean(axis=0))
    center_term = min(center_distance / float(weights["max_center_distance"]), 1.0)
    similarity = pose_similarity(previous, current, config)
    pose_term = 1.0 - similarity
    iou_term = 1.0 - box_iou(keypoints_to_box(previous), detection_box)
    return (
        float(weights["center_weight"]) * center_term
        + float(weights["iou_weight"]) * iou_term
        + float(weights["pose_weight"]) * pose_term
    )


def pose_similarity(previous, current, config: dict) -> float:
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    threshold = float(config["preprocess"]["conf_threshold"])
    previous_valid = _point_valid(previous, threshold)
    current_valid = _point_valid(current, threshold)
    comparable = previous_valid & current_valid
    if not np.any(comparable):
        return 0.0
    mouth = int(config["keypoints"]["mouth_index"])
    tail = int(config["keypoints"]["tail_index"])
    eps = float(config["preprocess"]["eps"])
    previous_norm = _scale_normalize(previous[:, :2], previous_valid, mouth, tail, eps)
    current_norm = _scale_normalize(current[:, :2], current_valid, mouth, tail, eps)
    distance = np.linalg.norm(previous_norm[comparable] - current_norm[comparable], axis=-1).mean()
    similarity = float(np.exp(-distance))
    return min(max(similarity, 0.0), 1.0)


def _point_valid(keypoints: np.ndarray, threshold: float) -> np.ndarray:
    xy_ok = np.isfinite(keypoints[:, :2]).all(axis=-1)
    if keypoints.shape[1] < 3:
        return xy_ok
    confidence = keypoints[:, 2]
    return xy_ok & np.isfinite(confidence) & (confidence >= threshold)


def _scale_normalize(xy: np.ndarray, valid: np.ndarray, mouth: int, tail: int, eps: float) -> np.ndarray:
    center = xy[valid].mean(axis=0) if np.any(valid) else xy.mean(axis=0)
    length = np.linalg.norm(xy[mouth] - xy[tail])
    length = max(float(length), eps)
    return (xy - center) / length


def associate_detections(frames, config: dict) -> list:
    max_missing = int(config["association"]["max_missing_frames"])
    max_cost = float(config["association"]["max_cost"])
    active = []
    finished = []
    next_id = 1
    for frame_number, detections in frames:
        if active:
            cost = np.zeros((len(active), max(len(detections), 1)), dtype=np.float64)
            if not detections:
                cost[:, 0] = max_cost + 1.0
            else:
                cost = np.zeros((len(active), len(detections)), dtype=np.float64)
                for row, track in enumerate(active):
                    for col, detection in enumerate(detections):
                        cost[row, col] = association_cost(
                            track["last"], detection["keypoints"], detection["box"], config
                        )
            matched_tracks = set()
            matched_detections = set()
            if detections and active:
                rows, cols = linear_sum_assignment(cost)
                for row, col in zip(rows, cols):
                    if cost[row, col] > max_cost:
                        continue
                    track = active[row]
                    detection = detections[col]
                    track["frames"].append((frame_number, detection["keypoints"]))
                    track["last"] = detection["keypoints"]
                    track["missing"] = 0
                    matched_tracks.add(row)
                    matched_detections.add(col)
            survivors = []
            for row, track in enumerate(active):
                if row in matched_tracks:
                    survivors.append(track)
                    continue
                track["missing"] += 1
                track["frames"].append((frame_number, None))
                if track["missing"] > max_missing:
                    finished.append(track)
                else:
                    survivors.append(track)
            active = survivors
            for col, detection in enumerate(detections):
                if col in matched_detections:
                    continue
                active.append(_new_track(next_id, frame_number, detection))
                next_id += 1
        else:
            for detection in detections:
                active.append(_new_track(next_id, frame_number, detection))
                next_id += 1
    finished.extend(active)
    return finished


def _new_track(track_id: int, frame_number: int, detection: dict) -> dict:
    return {
        "track_id": track_id,
        "frames": [(frame_number, detection["keypoints"])],
        "last": detection["keypoints"],
        "missing": 0,
    }


def densify_track(frames, num_keypoints: int, max_gap: int) -> list:
    ordered = sorted(frames, key=lambda item: item[0])
    while ordered and ordered[0][1] is None:
        ordered.pop(0)
    while ordered and ordered[-1][1] is None:
        ordered.pop()
    if not ordered:
        return []
    segments = [[ordered[0]]]
    for item in ordered[1:]:
        gap = item[0] - segments[-1][-1][0] - 1
        if gap > max_gap:
            segments.append([item])
            continue
        previous = segments[-1][-1][0]
        for step in range(1, gap + 1):
            segments[-1].append((previous + step, None))
        segments[-1].append(item)
    limited = []
    for segment in segments:
        limited.extend(_split_long_missing(segment, max_gap))
    dense = []
    for segment in limited:
        if any(keypoints is not None for _, keypoints in segment):
            dense.append((int(segment[0][0]), _interpolate_segment(segment, num_keypoints)))
    return dense


def _split_long_missing(segment, max_gap: int) -> list:
    parts = []
    current = []
    pending = []
    for item in segment:
        if item[1] is None:
            pending.append(item)
            continue
        if len(pending) > max_gap:
            if current:
                parts.append(current)
            current = [item]
        else:
            current.extend(pending)
            current.append(item)
        pending = []
    if current:
        parts.append(current)
    return parts


def _interpolate_segment(frames, num_keypoints: int) -> np.ndarray:
    known = [(index, keypoints) for index, (_, keypoints) in enumerate(frames) if keypoints is not None]
    if not known:
        return np.zeros((0, num_keypoints, 3), dtype=np.float32)
    start = known[0][0]
    end = known[-1][0]
    sequence = np.zeros((end - start + 1, num_keypoints, 3), dtype=np.float32)
    known_positions = []
    known_values = []
    for index, keypoints in known:
        if start <= index <= end:
            known_positions.append(index - start)
            value = np.asarray(keypoints, dtype=np.float32)
            if value.shape[-1] == 2:
                padded = np.ones((num_keypoints, 3), dtype=np.float32)
                padded[:, :2] = value[:, :2]
                value = padded
            known_values.append(value)
    known_positions = np.asarray(known_positions, dtype=np.float64)
    known_values = np.stack(known_values, axis=0)
    timeline = np.arange(sequence.shape[0], dtype=np.float64)
    for keypoint_id in range(num_keypoints):
        for channel in range(3):
            sequence[:, keypoint_id, channel] = np.interp(
                timeline, known_positions, known_values[:, keypoint_id, channel]
            )
    for index, (_, keypoints) in enumerate(frames):
        if keypoints is None and start <= index <= end:
            sequence[index - start, :, 2] = 0.0
    return sequence


def jump_frame_ratio(sequence: np.ndarray, config: dict) -> float:
    if len(sequence) < 2:
        return 0.0
    jumps = 0
    for frame_id in range(1, len(sequence)):
        if _is_outlier_jump(sequence[frame_id - 1], sequence[frame_id], config):
            jumps += 1
    return jumps / (len(sequence) - 1)


def window_is_reliable(sequence: np.ndarray, config: dict) -> bool:
    array = np.asarray(sequence)
    if array.ndim != 3 or array.shape[0] == 0:
        return False
    confidence = array[:, :, 2] if array.shape[-1] >= 3 else np.ones(array.shape[:2], dtype=np.float32)
    rules = config["association"]
    threshold = float(config["preprocess"]["conf_threshold"])
    min_points = int(rules["min_valid_keypoints"])
    effective = (confidence >= threshold).sum(axis=1) >= min_points
    if float(np.mean(effective)) < float(rules["min_valid_ratio"]):
        return False
    if float(confidence.mean()) < float(rules["min_mean_confidence"]):
        return False
    if _longest_false_run(effective) > int(rules["max_consecutive_missing"]):
        return False
    if jump_frame_ratio(array, config) >= float(rules["max_jump_frame_ratio"]):
        return False
    return True


def _longest_false_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _is_outlier_jump(previous, current, config: dict) -> bool:
    prev_xy = np.asarray(previous)[:, :2]
    curr_xy = np.asarray(current)[:, :2]
    displacement = np.linalg.norm(curr_xy - prev_xy, axis=-1)
    mouth = int(config["keypoints"]["mouth_index"])
    tail = int(config["keypoints"]["tail_index"])
    eps = float(config["preprocess"]["eps"])
    body_length = max(float(np.linalg.norm(prev_xy[mouth] - prev_xy[tail])), eps)
    limit = float(config["association"]["max_jump_body_lengths"]) * body_length
    median = float(np.median(displacement))
    peak = float(np.max(displacement))
    return peak > limit and peak > 3.0 * max(median, eps)


def suppress_jumps(sequence: np.ndarray, config: dict) -> np.ndarray:
    if len(sequence) < 2:
        return sequence
    output = np.array(sequence, dtype=np.float32, copy=True)
    mouth = int(config["keypoints"]["mouth_index"])
    tail = int(config["keypoints"]["tail_index"])
    eps = float(config["preprocess"]["eps"])
    limit = float(config["association"]["max_jump_body_lengths"])
    for frame_id in range(1, output.shape[0]):
        body_length = np.linalg.norm(output[frame_id - 1, mouth, :2] - output[frame_id - 1, tail, :2])
        body_length = max(float(body_length), eps)
        jump = np.linalg.norm(output[frame_id, :, :2] - output[frame_id - 1, :, :2], axis=-1)
        median = float(np.median(jump))
        bad = (jump > limit * body_length) & (jump > 3.0 * max(median, eps))
        output[frame_id, bad, :2] = output[frame_id - 1, bad, :2]
        output[frame_id, bad, 2] = 0.0
    return output
