"""
find_columns.py  -  auto-detect bubble column X positions from template.png
Run:  python find_columns.py
Outputs: TEMPLATE_COLUMNS to paste into app.py + debug_columns.png overlay
"""
import sys
import cv2
import numpy as np

# ── config ─────────────────────────────────────────────────────────────────────
WARP_W, WARP_H   = 1000, 1414
NUM_COL_GROUPS   = 5       # how many question-group columns on the sheet
NUM_OPTIONS      = 4       # A,B,C,D  (change to 5 if sheet has A-E)

# ── helpers ────────────────────────────────────────────────────────────────────
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)];  rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def find_square(thresh, x0, y0, x1, y1):
    region = thresh[y0:y1, x0:x1]
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, ba = None, 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 100 or a > 80000: continue
        x,y,w,h = cv2.boundingRect(c)
        if w < 10 or h < 10: continue
        if not (0.3 < w/float(h) < 3.0): continue
        ha = cv2.contourArea(cv2.convexHull(c))
        if ha < 1: continue
        if not (0.75 < a/ha <= 1.0): continue
        if a > ba:
            ba = a
            best = {"cx":x+w//2+x0,"cy":y+h//2+y0,"x":x+x0,"y":y+y0,"w":w,"h":h}
    return best

def warp_template(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh   = clahe.apply(gray)
    blur  = cv2.GaussianBlur(enh,(5,5),0)
    _,th  = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    h,w   = th.shape
    pts   = None
    for frac in (0.18,0.22,0.27,0.33,0.40):
        qw,qh = max(1,int(w*frac)), max(1,int(h*frac))
        tl = find_square(th,0,0,qw,qh)
        tr = find_square(th,w-qw,0,w,qh)
        bl = find_square(th,0,h-qh,qw,h)
        br = find_square(th,w-qw,h-qh,w,h)
        if tl and tr and bl and br:
            print(f"  Corner squares found at frac={frac}")
            pts = order_points(np.array([
                [tl["x"],          tl["y"]],
                [tr["x"]+tr["w"],  tr["y"]],
                [br["x"]+br["w"],  br["y"]+br["h"]],
                [bl["x"],          bl["y"]+bl["h"]]], dtype="float32"))
            break

    if pts is not None:
        tl_,tr_,br_,bl_ = pts
        mW = int(max(np.linalg.norm(br_-bl_),np.linalg.norm(tr_-tl_)))
        mH = int(max(np.linalg.norm(tr_-br_),np.linalg.norm(tl_-bl_)))
        dst = np.array([[0,0],[mW-1,0],[mW-1,mH-1],[0,mH-1]],dtype="float32")
        warped = cv2.warpPerspective(img,cv2.getPerspectiveTransform(pts,dst),(mW,mH))
        print("  Perspective warp applied")
    else:
        print("  WARNING: corners not found, using full image")
        warped = img.copy()

    return cv2.resize(warped,(WARP_W,WARP_H))

def cluster_1d(values, gap):
    if not values: return []
    vs = sorted(set(values))
    clusters, curr = [], [vs[0]]
    for v in vs[1:]:
        if v - curr[-1] > gap: clusters.append(curr); curr=[v]
        else: curr.append(v)
    clusters.append(curr)
    return clusters

# ── MAIN ───────────────────────────────────────────────────────────────────────
img = cv2.imread("template.png")
if img is None:
    sys.exit("ERROR: template.png not found. Save the new template as template.png first.")

print(f"Loaded template.png  {img.shape[1]}x{img.shape[0]}")
print("Warping...")
warped = warp_template(img)

# enhance after warp
gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_w = clahe.apply(gray_w)

# ── method 1: Hough circles ────────────────────────────────────────────────────
print("\nRunning Hough circle detection...")
blr = cv2.GaussianBlur(gray_w,(9,9),2)
for p2 in (15, 12, 10, 8):
    circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                param1=50, param2=p2, minRadius=4, maxRadius=16)
    if circles is not None and len(circles[0]) > 100:
        print(f"  param2={p2}  ->  {len(circles[0])} circles")
        break

hough_pts = []
if circles is not None:
    hough_pts = [(int(cx),int(cy)) for cx,cy,_ in np.round(circles[0]).astype(int)]

# ── method 2: contour circles (fallback + cross-check) ────────────────────────
print("Running contour detection...")
_,th2 = cv2.threshold(gray_w,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cnts,_ = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour_pts = []
for c in cnts:
    area = cv2.contourArea(c)
    if not (30 < area < 800): continue
    x,y,w,h = cv2.boundingRect(c)
    if not (0.5 < w/float(h) < 2.0): continue
    ha = cv2.contourArea(cv2.convexHull(c))
    if ha < 1: continue
    # Circles: solidity 0.6-0.9
    if not (0.55 < area/ha < 0.92): continue
    contour_pts.append((x+w//2, y+h//2))

print(f"  Contour method: {len(contour_pts)} bubble candidates")

# Combine both methods
all_pts = list(set(hough_pts + contour_pts))
print(f"  Combined: {len(all_pts)} unique candidates")

if len(all_pts) < 50:
    print("  Too few candidates — using raw contour pixels")
    all_pts = contour_pts if len(contour_pts) > len(hough_pts) else hough_pts

# ── cluster X values into columns ─────────────────────────────────────────────
all_xs  = sorted([p[0] for p in all_pts])
x_clust = cluster_1d(all_xs, gap=8)
# Keep clusters with enough circles to represent a real column (>=5 rows)
real_cols = [(int(np.mean(xc)), len(xc)) for xc in x_clust if len(xc) >= 5]
real_cols.sort()

print(f"\nX-columns with >=5 circles ({len(real_cols)} total):")
for i,(x,n) in enumerate(real_cols):
    print(f"  col {i+1:2d}: x={x:4d}  n={n}")

if not real_cols:
    sys.exit("No columns found. Check template.png is correct.")

# ── identify question-number columns vs bubble columns ────────────────────────
# Strategy: split into NUM_COL_GROUPS groups by large X-gaps,
# then within each group drop any column that's followed by an unusually large gap
# (the question-number column always has a bigger gap after it)

col_xs_all = [x for x,_ in real_cols]
if len(col_xs_all) > 1:
    all_gaps = np.diff(col_xs_all)
    med_gap  = np.median(all_gaps)
    print(f"\nMedian X gap between columns: {med_gap:.1f} px")

    # Find large gaps that separate the 5 major groups
    major_gap_threshold = med_gap * 1.8
    major_splits = [i for i,g in enumerate(all_gaps) if g > major_gap_threshold]
    print(f"Major group splits at column indices: {major_splits}")

    # Build groups
    groups = []
    start = 0
    for split_idx in major_splits:
        groups.append(col_xs_all[start:split_idx+1])
        start = split_idx + 1
    groups.append(col_xs_all[start:])

else:
    groups = [col_xs_all]

print(f"\n{len(groups)} column groups detected:")
for i,g in enumerate(groups):
    print(f"  Group {i+1}: X={g}")

# ── within each group, separate Que column from bubble columns ─────────────────
# The Que (question number) column is the LEFTMOST column in each group.
# It is followed by a LARGER internal gap (the number field is wider than a bubble).
bubble_groups = []
for g in groups:
    if len(g) <= 1:
        # Only 1 column found in this group — can't determine Que vs bubbles
        # Keep it as-is
        bubble_groups.append(g)
        continue

    int_gaps = np.diff(g)
    med_int  = np.median(int_gaps)

    # If the first gap is notably larger than the rest, first col = Que column
    if int_gaps[0] > med_int * 1.4:
        print(f"  Group {groups.index(g)+1}: dropping X={g[0]} (Que column — wider first gap {int_gaps[0]:.0f} vs median {med_int:.0f})")
        bubble_groups.append(list(g[1:]))
    else:
        bubble_groups.append(list(g))

# Keep only the first NUM_COL_GROUPS groups and NUM_OPTIONS columns per group
bubble_groups = bubble_groups[:NUM_COL_GROUPS]
bubble_groups = [g[:NUM_OPTIONS] for g in bubble_groups]

# ── print result ───────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("DETECTED BUBBLE COLUMN POSITIONS")
print("="*55)
labels = ["A","B","C","D","E"]
q_ranges = ["Q  1-40","Q 41-80","Q 81-120","Q121-160","Q161-200"]
print("\nPASTE THIS INTO app.py  ->  TEMPLATE_COLUMNS:")
print("TEMPLATE_COLUMNS = [")
for i,g in enumerate(bubble_groups):
    label = q_ranges[i] if i < len(q_ranges) else f"Group{i+1}"
    opt_labels = [f"{labels[j]}={x}" for j,x in enumerate(g)]
    print(f"    {g},   # {label}  ({', '.join(opt_labels)})")
print("]")

# ── save debug overlay ─────────────────────────────────────────────────────────
debug = warped.copy()

# All candidate circles  (light green)
for (cx,cy) in all_pts:
    cv2.circle(debug,(cx,cy),6,(0,200,0),1)

# Detected columns (blue vertical lines)
for x,_ in real_cols:
    cv2.line(debug,(x,0),(x,WARP_H),(255,120,0),1)

# Final bubble columns (red + option label)
col_colors = [(0,0,255),(0,100,255),(0,200,255),(0,255,200),(0,255,100)]
for g_idx, g in enumerate(bubble_groups):
    for o_idx, cx in enumerate(g):
        color = col_colors[o_idx % len(col_colors)]
        cv2.line(debug,(cx,0),(cx,WARP_H),color,2)
        # Label at top and every 200 rows
        for y_label in [20, 200, 400, 600, 800, 1000, 1200]:
            cv2.putText(debug, labels[o_idx],
                        (cx-6, y_label),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

cv2.imwrite("debug_columns.png", debug)
print("\nDebug image -> debug_columns.png")
print("Blue = all columns | Coloured lines = A,B,C,D per group")
