import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import itertools

app = FastAPI(title="OMR Scanner Universal API")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

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

        # Base thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

        # 1. FIND 4 X-MARKS
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x_marks = []
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < 30: continue
            
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            
            if 0.5 < aspect_ratio < 2.0:
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                
                solidity = area / float(hull_area)
                extent = area / float(w * h)
                
                # An 'X' or cross typically has low solidity (0.15-0.65) and extent (0.15-0.65)
                # whereas circles/squares have >0.75
                if 0.15 < solidity < 0.65 and 0.15 < extent < 0.65:
                    M = cv2.moments(c)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        x_marks.append((cx, cy))
        
        # 2. ALIGN VIEWPORT IF MARKERS FOUND
        if len(x_marks) >= 4:
            pts = np.array(x_marks, dtype="float32")
            if len(x_marks) > 4:
                max_area = 0
                best_pts = pts[:4]
                for combo in itertools.combinations(pts, 4):
                    ordered = order_points(np.array(combo, dtype="float32"))
                    area = cv2.contourArea(ordered)
                    if area > max_area:
                        max_area = area
                        best_pts = ordered
                pts = best_pts
            else:
                pts = order_points(pts)

            (tl, tr, br, bl) = pts
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(pts, dst)
            # Warp the image exactly to the 4 x-marks viewport!
            img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

        # 3. PRODUCTION-GRADE MULTI-PASS BUBBLE DETECTION
        output_img = img.copy()

        # === PASS 1: Adaptive threshold (good for empty bubble outlines) ===
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 5)
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # === PASS 2: Otsu threshold (good for filled dark bubbles) ===
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # === PASS 3: Fixed low threshold (catches very dark ink fills) ===
        _, thresh3 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        contours3, _ = cv2.findContours(thresh3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Merge all contour sets
        all_contours = list(contours1) + list(contours2) + list(contours3)

        # Extract all circle-like candidates from all passes
        candidates = []
        for c in all_contours:
            area = cv2.contourArea(c)
            if area < 30 or area > 15000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h)
            if ar < 0.5 or ar > 2.0:
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ > 0.55:
                cx = x + w // 2
                cy = y + h // 2
                r = max(w, h) // 2
                candidates.append((cx, cy, r, area))

        # Deduplicate overlapping detections (merge circles within proximity)
        merged = []
        used = set()
        for i, (cx1, cy1, r1, a1) in enumerate(candidates):
            if i in used:
                continue
            group = [(cx1, cy1, r1, a1)]
            for j in range(i + 1, len(candidates)):
                if j in used:
                    continue
                cx2, cy2, r2, a2 = candidates[j]
                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                # Merge if centers are within the larger radius (overlapping circles)
                if dist < max(r1, r2) * 1.2:
                    group.append((cx2, cy2, r2, a2))
                    used.add(j)
            used.add(i)
            # Average the center positions and pick the largest radius from the group
            avg_cx = int(np.mean([g[0] for g in group]))
            avg_cy = int(np.mean([g[1] for g in group]))
            max_r = max(g[2] for g in group)
            max_a = max(g[3] for g in group)
            merged.append((avg_cx, avg_cy, max_r, max_a))

        # Filter by BOUNDING BOX RADIUS — not contour area!
        # Radius is consistent between empty circles (ring outline) and filled circles (solid disc)
        # whereas contour area varies wildly (ring=small, disc=large)
        bubbles = []
        if merged:
            radii = [m[2] for m in merged]
            med_r = np.median(radii)
            # Keep anything within 0.4x to 2.5x the median radius
            filtered = [m for m in merged if 0.4 * med_r < m[2] < 2.5 * med_r]

            for (cx, cy, r, area) in filtered:
                # Use GRAYSCALE MEAN INTENSITY for mark detection
                # Sample a slightly smaller circle (80% radius) to avoid edge artifacts
                sample_r = max(int(r * 0.8), 1)
                mask = np.zeros(gray.shape, dtype='uint8')
                cv2.circle(mask, (cx, cy), sample_r, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]

                # Lower mean = darker = marked. Threshold at 180 (out of 255)
                marked = mean_val < 180

                bx, by = cx - r, cy - r
                bw, bh = r * 2, r * 2
                bubbles.append({
                    'rect': (bx, by, bw, bh),
                    'marked': marked,
                    'intensity': mean_val
                })

                color = (0, 255, 0) if marked else (0, 0, 255)
                thickness = 3 if marked else 1
                cv2.rectangle(output_img, (bx, by), (bx + bw, by + bh), color, thickness)
        
        marked_count = sum(1 for b in bubbles if b['marked'])
        unmarked_count = len(bubbles) - marked_count
        
        # ====== GRID-BASED QUESTION MAPPING ======
        # Map bubbles to questions by analyzing their spatial arrangement
        questions = {}
        if bubbles:
            option_labels = ['A', 'B', 'C', 'D', 'E']
            
            # Get bubble centers
            centers = []
            for b in bubbles:
                bx, by, bw, bh = b['rect']
                cx = bx + bw // 2
                cy = by + bh // 2
                centers.append({'cx': cx, 'cy': cy, 'marked': b['marked']})
            
            # Step 1: Cluster into rows by Y-coordinate
            centers.sort(key=lambda c: c['cy'])
            rows = []
            current_row = [centers[0]]
            # Use median bubble height as row tolerance
            all_heights = [b['rect'][3] for b in bubbles]
            row_tol = np.median(all_heights) * 0.8
            
            for c in centers[1:]:
                if abs(c['cy'] - current_row[0]['cy']) < row_tol:
                    current_row.append(c)
                else:
                    rows.append(sorted(current_row, key=lambda c: c['cx']))
                    current_row = [c]
            rows.append(sorted(current_row, key=lambda c: c['cx']))
            
            # Step 2: Within each row, split into question groups by X-gaps
            all_row_groups = []  # list of (row_idx, group) tuples
            for row_idx, row in enumerate(rows):
                if len(row) < 2:
                    all_row_groups.append((row_idx, row))
                    continue
                
                # Calculate gaps between consecutive bubbles
                gaps = [row[i+1]['cx'] - row[i]['cx'] for i in range(len(row) - 1)]
                median_gap = np.median(gaps)
                gap_threshold = median_gap * 1.8  # Gaps >1.8x median = column separator
                
                # Split into groups
                groups = []
                current_group = [row[0]]
                for i in range(len(gaps)):
                    if gaps[i] > gap_threshold:
                        groups.append(current_group)
                        current_group = [row[i + 1]]
                    else:
                        current_group.append(row[i + 1])
                groups.append(current_group)
                
                for g in groups:
                    all_row_groups.append((row_idx, g))
            
            # Step 3: Clean up groups — remove question numbers
            # Question numbers are text printed LEFT of actual bubble options.
            # Strategy: enforce exactly num_options bubbles per group.
            # If a group has more, pick the best contiguous run with most uniform spacing.
            cleaned_groups = []
            for row_idx, group in all_row_groups:
                if len(group) <= num_options:
                    cleaned_groups.append((row_idx, group))
                else:
                    # Too many elements — some are question numbers
                    # Find the best contiguous run of num_options elements
                    # with the most uniform spacing (real bubbles are evenly spaced)
                    best_start = len(group) - num_options  # default: rightmost
                    best_score = float('inf')
                    
                    for start in range(len(group) - num_options + 1):
                        subgroup = group[start:start + num_options]
                        sub_gaps = [subgroup[i+1]['cx'] - subgroup[i]['cx']
                                    for i in range(len(subgroup) - 1)]
                        score = np.var(sub_gaps)  # lower variance = more uniform
                        if score < best_score:
                            best_score = score
                            best_start = start
                    
                    cleaned_groups.append((row_idx, group[best_start:best_start + num_options]))
            
            # Step 4: Organize into columns
            group_info = []
            for row_idx, group in cleaned_groups:
                start_x = group[0]['cx']
                group_info.append({'row_idx': row_idx, 'start_x': start_x, 'group': group})
            
            # Sort by start_x, then cluster into columns
            group_info.sort(key=lambda g: g['start_x'])
            columns = []
            current_col = [group_info[0]]
            col_tol = np.median(all_heights) * 3
            
            for gi in group_info[1:]:
                if abs(gi['start_x'] - current_col[0]['start_x']) < col_tol:
                    current_col.append(gi)
                else:
                    columns.append(current_col)
                    current_col = [gi]
            columns.append(current_col)
            
            # Step 5: Within each column, sort by row, assign question numbers
            q_num = 1
            for col in columns:
                col.sort(key=lambda g: g['row_idx'])
                for gi in col:
                    group = gi['group']
                    marked_options = []
                    for idx, bubble in enumerate(group):
                        label = option_labels[idx] if idx < len(option_labels) else f"O{idx+1}"
                        if bubble['marked']:
                            marked_options.append(label)
                    
                    if len(marked_options) > 1:
                        answer = f"Rejected ({', '.join(marked_options)})"
                    elif len(marked_options) == 1:
                        answer = marked_options[0]
                    else:
                        answer = '—'
                    questions[f"Q{q_num}"] = answer
                    q_num += 1

        _, buffer = cv2.imencode('.jpg', output_img)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        b64_data = f"data:image/jpeg;base64,{b64_str}"
        
        data = {
            "Total Bubbles": len(bubbles),
            "Marked Bubbles": marked_count,
            "Unmarked Bubbles": unmarked_count
        }

        # Let the user know if perspective scan was active
        if len(x_marks) >= 4:
            data['Viewport Locked'] = 'Yes (Aligned to X marks)'
        else:
            data['Viewport Locked'] = 'No (X bounds omitted/not found)'

        return {"success": True, "data": data, "questions": questions, "image": b64_data}

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9004)
