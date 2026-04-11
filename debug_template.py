import cv2
import numpy as np

def detect_x_candidates(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100 or area > 5000: continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if not (0.6 < ar < 1.4): continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = area / hull_area
        if not (0.2 < solidity < 0.7): continue
        cx = x + w // 2
        cy = y + h // 2
        candidates.append((cx, cy))
    return candidates

def get_4_corners(points):
    pts = np.array(points)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

img = cv2.imread('template.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

candidates = detect_x_candidates(thresh)
print(f"Detected {len(candidates)} X markers")

pts = None
if len(candidates) >= 4:
    pts = get_4_corners(candidates)

if pts is not None:
    pts = order_points(pts)
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
else:
    warped = img

warped = cv2.resize(warped, (1000, 1600))
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# detect row markers
_, t = cv2.threshold(warped_gray, 50, 255, cv2.THRESH_BINARY_INV)
h, w = warped_gray.shape
left_strip = t[:, :int(0.15 * w)]
contours, _ = cv2.findContours(left_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
markers = []
for c in contours:
    x, y, wc, hc = cv2.boundingRect(c)
    if 5 < wc < 80 and 5 < hc < 25:
        cy = y + hc // 2
        markers.append(cy)
markers = sorted(markers)
cleaned = []
for m in markers:
    if not cleaned or abs(m - cleaned[-1]) > 10:
        cleaned.append(m)

print(f"Detected {len(cleaned)} rows")

# Now detect all bubbles in the warped document
cnts, _ = cv2.findContours(t, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
bubbles = []
for c in cnts:
    x, y, wc, hc = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    ar = wc / float(hc)
    if 0.5 < ar < 1.5 and 50 < area < 400:
        bubbles.append((x + wc//2, y + hc//2, wc))

columns = {}
for bx, by, bw in bubbles:
    # cluster by X coordinate (column)
    found_col = False
    for cx in columns.keys():
        if abs(cx - bx) < 15:
            columns[cx].append(by)
            found_col = True
            break
    if not found_col:
        columns[bx] = [by]

print(f"Detected {len(columns)} X clusters (columns)")
print("X coordinates:", sorted(columns.keys()))
