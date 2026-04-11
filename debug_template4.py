import cv2
import numpy as np
import json

def get_template_config():
    img = cv2.imread('template.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Process exactly like the production pipeline
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 1. 4 X-Corners
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if 100 < area < 5000:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h)
            if 0.6 < ar < 1.4:
                hull = cv2.convexHull(c)
                if cv2.contourArea(hull) > 0 and (area / cv2.contourArea(hull)) > 0.2:
                    candidates.append((x + w//2, y + h//2))
                    
    pts = None
    if len(candidates) >= 4:
        pts_arr = np.array(candidates)
        s = pts_arr.sum(axis=1)
        diff = np.diff(pts_arr, axis=1)
        pts = np.array([pts_arr[np.argmin(s)], pts_arr[np.argmin(diff)], 
                        pts_arr[np.argmax(s)], pts_arr[np.argmax(diff)]], dtype="float32")
    
    if pts is not None:
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
    
    # Find circles
    _, thresh_w = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_w, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        area = cv2.contourArea(c)
        if 50 < area < 600 and 0.5 < ar < 1.5:
            bubbles.append({"x": x + w//2, "y": y + h//2, "w": w})
            
    # Group by X to find columns
    x_coords = [b['x'] for b in bubbles]
    hist, bins = np.histogram(x_coords, bins=1000, range=(0, 1000))
    # Find peaks which are > 10
    peaks = []
    for i in range(len(hist)):
        if hist[i] > 10:
            peaks.append(int(bins[i]))
            
    # Clean up peaks (merge close ones)
    clean_peaks = []
    for p in peaks:
        if not clean_peaks or p - clean_peaks[-1] > 10:
            clean_peaks.append(p)
    return clean_peaks

print(get_template_config())
