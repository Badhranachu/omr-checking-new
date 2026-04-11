import cv2
import numpy as np

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
        if area < 100 or area > 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if not (0.6 < ar < 1.4): continue 
        hull=cv2.convexHull(c) 
        hull_area=cv2.contourArea(hull) 
        if hull_area==0: continue
        solidity=area / hull_area 
        if not (0.2 < solidity < 0.7): continue 
        cx=x + w // 2 
        cy=y + h // 2
        candidates.append((cx, cy)) 
    return candidates 

def get_4_corners(points): 
    pts=np.array(points) 
    s=pts.sum(axis=1)
    diff=np.diff(pts, axis=1) 
    tl=pts[np.argmin(s)] 
    br=pts[np.argmax(s)] 
    tr=pts[np.argmin(diff)]
    bl=pts[np.argmax(diff)] 
    return np.array([tl, tr, br, bl], dtype="float32" ) 
    
def validate_quad(pts): 
    (tl, tr, br, bl)=pts 
    width_top=np.linalg.norm(tr - tl) 
    width_bottom=np.linalg.norm(br - bl) 
    height_left=np.linalg.norm(bl - tl) 
    height_right=np.linalg.norm(br - tr) 
    if abs(width_top - width_bottom)> 0.2 * width_top:
        return False
    if abs(height_left - height_right) > 0.2 * height_left:
        return False
    if width_top < 400 or height_left < 600: return False 
    return True 
    
def fallback_document(thresh): 
    contours, _=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    if not contours: return None
    largest=max(contours, key=cv2.contourArea) 
    peri=cv2.arcLength(largest, True)
    approx=cv2.approxPolyDP(largest, 0.02 * peri, True) 
    if len(approx)==4: 
        return order_points(approx.reshape(4, 2)) 
    return None 

def detect_row_markers(gray): 
    _, thresh=cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV) 
    h, w=gray.shape
    left_strip=thresh[:, :int(0.15 * w)] 
    contours, _=cv2.findContours(left_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    markers=[] 
    for c in contours: 
        x, y, wc, hc=cv2.boundingRect(c) 
        if 5 < wc < 80 and 5 < hc < 25: 
            cy=y + hc // 2 
            markers.append(cy) 
    markers=sorted(markers) 
    
    cleaned=[] 
    for m in markers: 
        if not cleaned or abs(m - cleaned[-1])> 10:
            cleaned.append(m)
    return cleaned

def scan_omr_test(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    candidates = detect_x_candidates(thresh)
    pts = None
    if len(candidates) >= 4:
        pts_candidate = get_4_corners(candidates)
        if validate_quad(pts_candidate):
            pts = pts_candidate

    if pts is None:
        pts = fallback_document(thresh)

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

    warped = cv2.resize(warped, (1000, 1600))
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    markers = detect_row_markers(gray_w)
    return markers

print('Processing template.png')
print('Markers found:', scan_omr_test('template.png'))
