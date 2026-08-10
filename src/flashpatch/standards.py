from __future__ import annotations

import math
from typing import Any


BT709_RELATIVE_LUMINANCE_COEFFICIENTS = (0.2126, 0.7152, 0.0722)
WCAG_SMALL_SAFE_AREA_PIXELS = 21_824


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def general_flash_transition_is_flash(first_luminance: float, second_luminance: float) -> bool:
    first = _finite_number(first_luminance, "first_luminance")
    second = _finite_number(second_luminance, "second_luminance")
    if not 0.0 <= first <= 1.0 or not 0.0 <= second <= 1.0:
        raise ValueError("relative luminance must be in [0, 1]")
    delta = abs(second - first)
    reaches_delta = delta >= 0.1 or math.isclose(
        delta, 0.1, rel_tol=0.0, abs_tol=4.0 * math.ulp(1.0)
    )
    return reaches_delta and min(first, second) < 0.8


def saturated_red_threshold_is_met(red_ratio: float, chroma_distance: float) -> bool:
    ratio = _finite_number(red_ratio, "red_ratio")
    distance = _finite_number(chroma_distance, "chroma_distance")
    if not 0.0 <= ratio <= 1.0 or distance < 0.0:
        raise ValueError("red ratio must be in [0, 1] and chroma distance must be non-negative")
    return ratio >= 0.8 and distance > 0.2


def _rgb_metrics(rgb: tuple[int, int, int]) -> tuple[float, tuple[float, float]]:
    if len(rgb) != 3 or any(isinstance(value, bool) or not isinstance(value, int) for value in rgb):
        raise ValueError("RGB values must be three 8-bit integers")
    if any(value < 0 or value > 255 for value in rgb):
        raise ValueError("RGB values must be three 8-bit integers")
    encoded = tuple(value / 255.0 for value in rgb)
    total = sum(encoded)
    red_ratio = encoded[0] / total if total else 0.0
    linear = tuple(
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in encoded
    )
    x = 0.4124 * linear[0] + 0.3576 * linear[1] + 0.1805 * linear[2]
    y = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    z = 0.0193 * linear[0] + 0.1192 * linear[1] + 0.9505 * linear[2]
    denominator = x + 15.0 * y + 3.0 * z
    chromaticity = (4.0 * x / denominator, 9.0 * y / denominator) if denominator else (0.0, 0.0)
    return red_ratio, chromaticity


def red_flash_transition_is_flash(
    first_rgb: tuple[int, int, int], second_rgb: tuple[int, int, int]
) -> bool:
    first_ratio, first_uv = _rgb_metrics(first_rgb)
    second_ratio, second_uv = _rgb_metrics(second_rgb)
    distance = math.hypot(first_uv[0] - second_uv[0], first_uv[1] - second_uv[1])
    return saturated_red_threshold_is_met(max(first_ratio, second_ratio), distance)


def wcag_small_area_exemption_applies(contiguous_pixels: int) -> bool:
    if isinstance(contiguous_pixels, bool) or not isinstance(contiguous_pixels, int):
        raise ValueError("contiguous_pixels must be a non-negative integer")
    if contiguous_pixels < 0:
        raise ValueError("contiguous_pixels must be a non-negative integer")
    return contiguous_pixels < WCAG_SMALL_SAFE_AREA_PIXELS


def wcag_frequency_path_passes(flash_count: int, *, same_state_at_window_ends: bool) -> bool:
    if isinstance(flash_count, bool) or not isinstance(flash_count, int) or flash_count < 0:
        raise ValueError("flash_count must be a non-negative integer")
    if not isinstance(same_state_at_window_ends, bool):
        raise ValueError("same_state_at_window_ends must be boolean")
    return flash_count < 3 or (flash_count == 3 and same_state_at_window_ends)


def itu_luminance_transition_is_flash(
    first_cd_m2: float,
    second_cd_m2: float,
    *,
    michelson_inclusive: bool = False,
) -> bool:
    first = _finite_number(first_cd_m2, "first_cd_m2")
    second = _finite_number(second_cd_m2, "second_cd_m2")
    if first < 0.0 or second < 0.0:
        raise ValueError("absolute luminance must be non-negative")
    darker, lighter = sorted((first, second))
    if darker < 160.0:
        return lighter - darker >= 20.0
    contrast = (lighter - darker) / (lighter + darker)
    return contrast >= 1.0 / 17.0 if michelson_inclusive else contrast > 1.0 / 17.0


def regular_pattern_is_hazardous(
    *,
    stripe_pairs: int,
    affected_fraction: float,
    dynamic: bool,
    smooth_one_direction: bool = False,
) -> bool:
    if isinstance(stripe_pairs, bool) or not isinstance(stripe_pairs, int) or stripe_pairs < 0:
        raise ValueError("stripe_pairs must be a non-negative integer")
    fraction = _finite_number(affected_fraction, "affected_fraction")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("affected_fraction must be in [0, 1]")
    if not isinstance(dynamic, bool) or not isinstance(smooth_one_direction, bool):
        raise ValueError("dynamic and smooth_one_direction must be boolean")
    if smooth_one_direction:
        return False
    area_threshold = 0.25 if dynamic else 0.40
    return stripe_pairs > 5 and fraction > area_threshold


def normalize_color_profile(profile: Any) -> str:
    if not isinstance(profile, str) or profile.strip().lower() not in {"srgb", "bt709"}:
        raise ValueError("the SDR profile requires explicit sRGB/BT.709 color metadata")
    return "srgb-bt709"


def evaluate_boundary_vector(vector: dict[str, Any]) -> bool | str:
    operation = vector.get("operation")
    inputs = vector.get("input")
    if not isinstance(inputs, dict):
        raise ValueError("boundary vector input must be an object")
    if operation == "general_flash_transition":
        return general_flash_transition_is_flash(inputs["first_luminance"], inputs["second_luminance"])
    if operation == "saturated_red_threshold":
        return saturated_red_threshold_is_met(inputs["red_ratio"], inputs["chroma_distance"])
    if operation == "wcag_small_area_exemption":
        return wcag_small_area_exemption_applies(inputs["contiguous_pixels"])
    if operation == "wcag_frequency_path":
        return wcag_frequency_path_passes(
            inputs["flash_count"], same_state_at_window_ends=inputs["same_state_at_window_ends"]
        )
    if operation == "itu_luminance_transition":
        return itu_luminance_transition_is_flash(
            inputs["first_cd_m2"],
            inputs["second_cd_m2"],
            michelson_inclusive=inputs.get("michelson_inclusive", False),
        )
    if operation == "regular_pattern":
        return regular_pattern_is_hazardous(**inputs)
    if operation == "color_profile":
        try:
            return normalize_color_profile(inputs["profile"])
        except ValueError:
            return "rejected"
    raise ValueError(f"unsupported boundary-vector operation: {operation}")
