import cv2
import numpy as np

def detect_x_candidates(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100 or area > 50000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if not (0.6 < ar < 1.4): continue 
        hull = cv2.convexHull(c) 
        if cv2.contourArea(hull) == 0: continue
        solidity = area / cv2.contourArea(hull) 
        if not (0.2 < solidity < 0.7): continue 
        cx = x + w // 2 
        cy = y + h // 2
        candidates.append({'cx': cx, 'cy': cy, 'x': x, 'y': y, 'w': w, 'h': h}) 
    return candidates 

def get_4_corners(candidates): 
    centers = np.array([[c['cx'], c['cy']] for c in candidates])
    s = centers.sum(axis=1)
    diff = np.diff(centers, axis=1) 
    
    tl_c = candidates[np.argmin(s)]
    br_c = candidates[np.argmax(s)]
    tr_c = candidates[np.argmin(diff)]
    bl_c = candidates[np.argmax(diff)]
    
    tl = [tl_c['x'], tl_c['y']]
    br = [br_c['x'] + br_c['w'], br_c['y'] + br_c['h']]
    tr = [tr_c['x'] + tr_c['w'], tr_c['y']]
    bl = [bl_c['x'], bl_c['y'] + bl_c['h']]
    
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
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth, maxHeight = int(max(widthA, widthB)), int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))
else:
    warped = gray
    
warped = cv2.resize(warped, (1000, 1600))
_, thresh_w = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh_w, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
bubbles = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)
    area = cv2.contourArea(c)
    if 50 < area < 600 and 0.5 < ar < 1.5:
        bubbles.append(x + w//2)

hist, bins = np.histogram(bubbles, bins=1000, range=(0, 1000))
peaks = []
for i in range(len(hist)):
    if hist[i] > 10:
        peaks.append(int(bins[i]))
        
clean_peaks = []
for p in peaks:
    if not clean_peaks or p - clean_peaks[-1] > 10:
        clean_peaks.append(p)
print(clean_peaks)
