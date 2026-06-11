from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class RelationPlannerConfig:
    enabled: bool = False
    d_ref_mm: float = 2.0
    jump_mm: float = 2.5
    illegal_jump_mm: float = 18.0
    big_m: float = 1_000_000.0
    w_d: float = 1.0
    w_jump: float = 1.2
    w_trim: float = 0.8
    w_lock: float = 0.2
    w_near_connect: float = 1.0
    w_axis_align: float = 0.5
    w_endpoint_compat: float = 0.6
    w_retrieval: float = 0.8
    trim_mm: float = 10.0
    near_connect_scale_mm: float = 2.5
    endpoint_compat_mm: float = 3.0
    retrieval_prior: float = 0.0


def _parse_scalar(value: str) -> object:
    raw = value.strip()
    if raw.lower() in {"true", "yes", "on"}:
        return True
    if raw.lower() in {"false", "no", "off"}:
        return False
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip("\"'")


def load_relation_config(path: str | Path | None) -> RelationPlannerConfig:
    if not path:
        return RelationPlannerConfig()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Planner config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload: dict[str, object] = {}
        for line in text.splitlines():
            stripped = line.split("#", 1)[0].strip()
            if not stripped or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            payload[key.strip()] = _parse_scalar(value)
    allowed = set(RelationPlannerConfig.__dataclass_fields__)
    clean = {key: value for key, value in payload.items() if key in allowed}
    return RelationPlannerConfig(**clean)


def config_to_dict(config: RelationPlannerConfig | None) -> dict[str, object] | None:
    return asdict(config) if config is not None else None


def endpoint_direction(coords: list[tuple[float, float]], reverse: bool) -> np.ndarray:
    if len(coords) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    if reverse:
        a = np.array(coords[-1], dtype=np.float32)
        b = np.array(coords[-2], dtype=np.float32)
    else:
        a = np.array(coords[0], dtype=np.float32)
        b = np.array(coords[1], dtype=np.float32)
    vec = b - a
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def transition_direction(start_xy: tuple[float, float], end_xy: tuple[float, float]) -> np.ndarray:
    vec = np.array([end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def relation_transition_cost(
    distance_mm: float,
    near_signal: float,
    jump_signal: float,
    transition_dir: np.ndarray,
    segment_dir: np.ndarray,
    config: RelationPlannerConfig,
) -> tuple[float, dict[str, float]]:
    if not config.enabled:
        return distance_mm, {"distance_mm": float(distance_mm), "enabled": 0.0}
    if distance_mm >= config.illegal_jump_mm:
        return config.big_m, {"distance_mm": float(distance_mm), "illegal_jump": 1.0}

    normalized_distance = min(distance_mm / max(1e-6, config.d_ref_mm), 3.0)
    axis_align = float(abs(np.dot(transition_dir, segment_dir)))
    endpoint_compat = max(0.0, 1.0 - distance_mm / max(1e-6, config.endpoint_compat_mm))
    near_bonus = max(float(near_signal), max(0.0, 1.0 - distance_mm / max(1e-6, config.near_connect_scale_mm)))
    jump_risk = max(float(jump_signal), 1.0 if distance_mm >= config.jump_mm else 0.0)
    trim_risk = 1.0 if distance_mm >= config.trim_mm else 0.0
    lock_risk = 1.0 if distance_mm >= config.jump_mm else 0.0

    penalty = (
        config.w_d * normalized_distance
        + config.w_jump * jump_risk
        + config.w_trim * trim_risk
        + config.w_lock * lock_risk
    )
    bonus = (
        config.w_near_connect * near_bonus
        + config.w_axis_align * axis_align
        + config.w_endpoint_compat * endpoint_compat
        + config.w_retrieval * config.retrieval_prior
    )
    cost = penalty - bonus
    return float(cost), {
        "distance_mm": float(distance_mm),
        "normalized_distance": float(normalized_distance),
        "near_bonus": float(near_bonus),
        "jump_risk": float(jump_risk),
        "trim_risk": float(trim_risk),
        "axis_align": float(axis_align),
        "endpoint_compat": float(endpoint_compat),
        "retrieval_prior": float(config.retrieval_prior),
        "cost": float(cost),
    }
