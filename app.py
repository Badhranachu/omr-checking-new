import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI(title="OMR Scanner Universal API")

# Template configuration dynamically extracted
TEMPLATE_COLUMNS = [
    [75, 102, 129, 157, 184],
    [265, 293, 320, 347, 374],
    [456, 483, 511, 538, 565],
    [646, 674, 701, 728, 755],
    [837, 864, 891, 918, 945]
]

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def detect_x_candidates(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100 or area > 50000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if not (0.6 < ar < 1.4):
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if not (0.2 < solidity < 0.7):
            continue
        cx = x + w // 2
        cy = y + h // 2
        candidates.append({"cx": cx, "cy": cy, "x": x, "y": y, "w": w, "h": h})
    return candidates

def get_4_corners(candidates):
    centers = np.array([[c["cx"], c["cy"]] for c in candidates])
    s = centers.sum(axis=1)
    diff = np.diff(centers, axis=1)

    tl_c = candidates[np.argmin(s)]
    br_c = candidates[np.argmax(s)]
    tr_c = candidates[np.argmin(diff)]
    bl_c = candidates[np.argmax(diff)]

    tl = [tl_c["x"], tl_c["y"]]
    br = [br_c["x"] + br_c["w"], br_c["y"] + br_c["h"]]
    tr = [tr_c["x"] + tr_c["w"], tr_c["y"]]
    bl = [bl_c["x"], bl_c["y"] + bl_c["h"]]

    return np.array([tl, tr, br, bl], dtype="float32")

def validate_quad(pts):
    (tl, tr, br, bl) = pts
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    if abs(width_top - width_bottom) > 0.2 * width_top:
        return False
    if abs(height_left - height_right) > 0.2 * height_left:
        return False
    if width_top < 400 or height_left < 600:
        return False
    return True

def fallback_document(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) == 4:
        return order_points(approx.reshape(4, 2))
    return None

def detect_row_markers(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = thresh.shape
    left_strip = thresh[:, :int(0.15 * w)]
    contours, _ = cv2.findContours(left_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []
    for c in contours:
        x, y, wc, hc = cv2.boundingRect(c)
        if 5 < wc < 100 and 5 < hc < 40:
            ar = wc / float(hc)
            if 1.0 < ar < 10.0:
                cy = y + hc // 2
                markers.append(cy)
    markers = sorted(markers)

    cleaned = []
    for m in markers:
        if not cleaned or abs(m - cleaned[-1]) > 10:
            cleaned.append(m)

    # Filter by consecutive uniform gaps
    if len(cleaned) > 2:
        gaps = np.diff(cleaned)
        med_gap = np.median(gaps)
        seqs = []
        curr_seq = [cleaned[0]]
        for i in range(len(gaps)):
            if abs(gaps[i] - med_gap) < med_gap * 0.4:
                curr_seq.append(cleaned[i + 1])
            else:
                seqs.append(curr_seq)
                curr_seq = [cleaned[i + 1]]
        seqs.append(curr_seq)

        longest_seq = max(seqs, key=len)

        # OMR Sheet Header Filter
        # "Questions start from second black strip only"
        if len(longest_seq) > 1:
            # Skip the 1st strip (header) and strictly take up to 40 questions
            return longest_seq[1:41]
        return longest_seq

    return cleaned

def build_circle_mask(shape, cx, cy, radius):
    mask = np.zeros(shape, dtype="uint8")
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask

def build_ring_mask(shape, cx, cy, inner_radius, outer_radius):
    mask = np.zeros(shape, dtype="uint8")
    cv2.circle(mask, (cx, cy), outer_radius, 255, -1)
    cv2.circle(mask, (cx, cy), inner_radius, 0, -1)
    return mask

def bubble_fill_metrics(gray, dark_mask, cx, cy):
    core_mask = build_circle_mask(gray.shape, cx, cy, 8)
    ring_mask = build_ring_mask(gray.shape, cx, cy, 8, 13)
    halo_mask = build_ring_mask(gray.shape, cx, cy, 13, 18)

    core_mean = cv2.mean(gray, mask=core_mask)[0]
    ring_mean = cv2.mean(gray, mask=ring_mask)[0]

    core_dark_ratio = cv2.mean(dark_mask, mask=core_mask)[0] / 255.0
    ring_dark_ratio = cv2.mean(dark_mask, mask=ring_mask)[0] / 255.0
    halo_dark_ratio = cv2.mean(dark_mask, mask=halo_mask)[0] / 255.0

    local_noise = max(ring_dark_ratio, halo_dark_ratio)
    contrast_score = max(0.0, ring_mean - core_mean) / 255.0
    fill_score = (core_dark_ratio - (0.55 * local_noise)) + (0.35 * contrast_score)

    return {
        "core_mean": core_mean,
        "ring_mean": ring_mean,
        "core_dark_ratio": core_dark_ratio,
        "ring_dark_ratio": ring_dark_ratio,
        "halo_dark_ratio": halo_dark_ratio,
        "fill_score": fill_score,
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/scan")
async def scan_omr(file: UploadFile = File(...), num_options: int = Form(4)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid image file provided."})

        # ==========================================
        # 1. CORE PREPROCESSING & PERSPECTIVE WARP
        # ==========================================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Robust X-marker fallback extraction
        candidates = detect_x_candidates(thresh)
        pts = None
        if len(candidates) >= 4:
            pts_candidate = get_4_corners(candidates)
            if validate_quad(pts_candidate):
                pts = pts_candidate

        fallback_used = False
        if pts is None:
            pts = fallback_document(thresh)
            fallback_used = True

        if pts is not None:
            pts = order_points(pts)
            (tl, tr, br, bl) = pts
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = int(max(heightA, heightB))
            dst = np.array([
                [0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
            ], dtype="float32")
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        else:
            warped = img

        warped = cv2.resize(warped, (1000, 1414))
        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur_w = cv2.GaussianBlur(gray_w, (5, 5), 0)
        _, dark_mask = cv2.threshold(blur_w, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ==========================================
        # 2. FAST TEMPLATE BUBBLE EVALUATION
        # ==========================================
        rows_y = detect_row_markers(gray_w)

        output_img = warped.copy()
        total_bubbles = 0
        marked_count = 0
        unmarked_count = 0

        # Draw structural row lines
        for y in rows_y:
            cv2.line(output_img, (0, y), (1000, y), (255, 0, 0), 1)

        q_num = 1
        option_labels = ["A", "B", "C", "D", "E"]
        bubble_samples = []

        # First pass: gather raw brightness intensity for every bubble
        y_offset = 0
        for col_x_coords in TEMPLATE_COLUMNS:
            x_offsets = col_x_coords[:num_options]
            for y_base in rows_y:
                y = y_base + y_offset
                for opt_idx, cx in enumerate(x_offsets):
                    total_bubbles += 1
                    sample_r = 6
                    mask = np.zeros(gray_w.shape, dtype='uint8')
                    cv2.circle(mask, (cx, y), sample_r, 255, -1)
                    mean_val = cv2.mean(gray_w, mask=mask)[0]
                    
                    bubble_samples.append({
                        "question": q_num,
                        "option": option_labels[opt_idx],
                        "cx": cx,
                        "cy": y,
                        "mean_val": mean_val
                    })
                q_num += 1

        # Second pass: Statistical thresholding
        all_intensities = np.array([sample["mean_val"] for sample in bubble_samples])
        if len(all_intensities) > 0:
            # "consider most repeated value unmarked"
            rounded_intensities = np.clip(np.round(all_intensities), 0, 255).astype(int)
            counts = np.bincount(rounded_intensities)
            unmarked_baseline = np.argmax(counts)
            # "any value which is atleast 30% more darker than that value is considered as marked"
            threshold_val = unmarked_baseline * 0.70
        else:
            unmarked_baseline = 190
            threshold_val = 180

        # Build results and draw debug layers
        questions = {}
        for idx in range(0, len(bubble_samples), num_options):
            group = bubble_samples[idx:idx + num_options]
            if not group: continue
            
            marked_options = []
            for sample in group:
                marked = sample["mean_val"] < threshold_val
                
                if marked:
                    marked_options.append(sample["option"])
                    marked_count += 1
                    color = (0, 255, 0)
                    thickness = 3
                else:
                    unmarked_count += 1
                    color = (0, 0, 255)
                    thickness = 2
                    
                cv2.circle(output_img, (sample["cx"], sample["cy"]), 14, color, thickness)
                
            question_num = group[0]["question"]
            if len(marked_options) > 1:
                questions[f"Q{question_num}"] = f"Rejected ({', '.join(marked_options)})"
            elif len(marked_options) == 1:
                questions[f"Q{question_num}"] = marked_options[0]
            else:
                questions[f"Q{question_num}"] = "—"

        _, buffer = cv2.imencode(".jpg", output_img)
        b64_str = base64.b64encode(buffer).decode("utf-8")
        b64_data = f"data:image/jpeg;base64,{b64_str}"

        status_str = "No (X bounds missed/Failed)"
        if pts is not None:
            if fallback_used:
                status_str = "Yes (Document Boundary Fallback)"
            else:
                status_str = "Yes (Aligned to X marks)"

        data = {
            "Total Bubbles": total_bubbles,
            "Marked Bubbles": marked_count,
            "Unmarked Bubbles": unmarked_count,
            "Viewport Locked": status_str,
            "Rows Detected": len(rows_y),
            "Mark Threshold": round(threshold_val, 3),
        }

        return {"success": True, "data": data, "questions": questions, "image": b64_data}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9004)
