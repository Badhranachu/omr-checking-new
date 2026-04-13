import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np


def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    return thresh


def _centroid(contour: np.ndarray) -> Tuple[int, int]:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w // 2, y + h // 2
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def detect_corner_marker_candidates(binary_inv: np.ndarray) -> List[Dict[str, float]]:
    h, w = binary_inv.shape[:2]
    img_area = float(h * w)
    min_area = max(150.0, img_area * 0.00002)
    max_area = img_area * 0.02

    contours, _ = cv2.findContours(binary_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Dict[str, float]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 10 or bh < 10:
            continue

        ar = bw / float(bh)
        if not (0.70 <= ar <= 1.35):
            continue

        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area <= 0:
            continue

        solidity = area / hull_area
        extent = area / float(max(1, bw * bh))
        if solidity < 0.82 or extent < 0.55:
            continue

        cx, cy = _centroid(c)

        # Must be near image corners, not center content.
        x_norm = min(cx / float(w), 1.0 - (cx / float(w)))
        y_norm = min(cy / float(h), 1.0 - (cy / float(h)))
        corner_proximity = x_norm + y_norm
        if corner_proximity > 0.70:
            continue

        candidates.append(
            {
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "cx": float(cx),
                "cy": float(cy),
                "area": float(area),
            }
        )

    return candidates


def select_and_order_4_markers(candidates: List[Dict[str, float]], image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    if len(candidates) < 4:
        raise ValueError(f"Marker detection failed: found {len(candidates)} candidate(s), need 4.")

    expected = {
        "tl": (0.0, 0.0),
        "tr": (w - 1.0, 0.0),
        "br": (w - 1.0, h - 1.0),
        "bl": (0.0, h - 1.0),
    }

    picked: Dict[str, Dict[str, float]] = {}
    used = set()

    # Pick best marker for each corner using distance + area score.
    for name in ("tl", "tr", "br", "bl"):
        ex, ey = expected[name]
        best_i = -1
        best_score = -1e18
        for i, c in enumerate(candidates):
            if i in used:
                continue
            dist = np.hypot(c["cx"] - ex, c["cy"] - ey)
            score = (2.0 * c["area"]) - (1.2 * dist)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i == -1:
            raise ValueError(f"Marker detection failed: could not assign {name}.")
        picked[name] = candidates[best_i]
        used.add(best_i)

    centers = np.array(
        [
            [picked["tl"]["cx"], picked["tl"]["cy"]],
            [picked["tr"]["cx"], picked["tr"]["cy"]],
            [picked["br"]["cx"], picked["br"]["cy"]],
            [picked["bl"]["cx"], picked["bl"]["cy"]],
        ],
        dtype=np.float32,
    )

    # Strict requested ordering:
    # tl: smallest x+y, br: largest x+y
    # tr/bl split by y-x (equivalent stable form of x-y rule)
    sums = centers.sum(axis=1)
    diffs = np.diff(centers, axis=1).reshape(-1)  # y - x

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = centers[np.argmin(sums)]   # top-left
    ordered[2] = centers[np.argmax(sums)]   # bottom-right
    ordered[1] = centers[np.argmin(diffs)]  # top-right
    ordered[3] = centers[np.argmax(diffs)]  # bottom-left

    return ordered


def warp_sheet(image: np.ndarray, ordered_points: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = ordered_points
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_w = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_h = int(max(height_right, height_left))

    if max_w < 50 or max_h < 50:
        raise ValueError("Perspective transform failed: computed warp dimensions are too small.")

    dst = np.array(
        [
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(ordered_points, dst)
    return cv2.warpPerspective(image, M, (max_w, max_h))


def correct_omr_perspective(input_path: str, output_path: str) -> str:
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    binary_inv = preprocess(image)
    candidates = detect_corner_marker_candidates(binary_inv)
    ordered_points = select_and_order_4_markers(candidates, image.shape)
    corrected = warp_sheet(image, ordered_points)

    if not cv2.imwrite(output_path, corrected):
        raise ValueError(f"Failed to write output image: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="OMR 4-marker perspective correction only.")
    parser.add_argument("input", help="Input OMR image path")
    parser.add_argument("--output", default="omr_corrected.jpg", help="Output aligned image path")
    args = parser.parse_args()

    out = correct_omr_perspective(args.input, args.output)
    print(f"Saved corrected image: {out}")


if __name__ == "__main__":
    main()
