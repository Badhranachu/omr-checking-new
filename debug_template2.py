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
cv2.imwrite('warped.jpg', warped)

gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
_, thresh_w = cv2.threshold(gray_w, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find ALL circles (bubbles) to get the X coordinates dynamically!
# In an OMR sheet, bubbles are highly uniform.
cnts, _ = cv2.findContours(thresh_w, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
bubbles = []
for c in cnts:
    area = cv2.contourArea(c)
    if 50 < area < 800:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if 0.5 < ar < 1.5:
            bubbles.append((x + w//2, y + h//2))

# print histogram of X coordinates to find the columns
x_coords = [b[0] for b in bubbles]
hist, bins = np.histogram(x_coords, bins=100, range=(0, 1000))
# Get peak bins
peaks = []
for i in range(len(hist)):
    if hist[i] > 10:  # at least 10 bubbles in this vertical column
        peaks.append(int(bins[i]))

print(f"X coordinate peaks: {peaks}")

# Also let's try the row detect with otsu threshold
left_strip = thresh_w[:, :int(0.15 * w)]
rcnts, _ = cv2.findContours(left_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
markers = []
for c in rcnts:
    x, y, wc, hc = cv2.boundingRect(c)
    if 5 < wc < 100 and 2 < hc < 30:
        cy = y + hc // 2
        markers.append(cy)

markers = sorted(markers)
cleaned = []
for m in markers:
    if not cleaned or abs(m - cleaned[-1]) > 8:
        cleaned.append(m)

print(f"Detected {len(cleaned)} rows. First rows: {cleaned[:10]}")
