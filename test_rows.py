import cv2
import numpy as np

def test_row_markers(image_path):
    img = cv2.imread(image_path)
    if img is None: return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
                markers.append(y + hc // 2)
                
    markers = sorted(markers)
    cleaned = []
    for m in markers:
        if not cleaned or abs(m - cleaned[-1]) > 10:
            cleaned.append(m)
            
    if len(cleaned) > 1:
        gaps = np.diff(cleaned)
        med_gap = np.median(gaps)
        print(f"Median gap: {med_gap}")
        print("Gaps:", gaps)
        
        # Filter out outliers at the beginning or end
        # We find the longest consecutive sequence of valid gaps
        seqs = []
        curr_seq = [cleaned[0]]
        for i in range(len(gaps)):
            if abs(gaps[i] - med_gap) < med_gap * 0.3:
                curr_seq.append(cleaned[i+1])
            else:
                seqs.append(curr_seq)
                curr_seq = [cleaned[i+1]]
        seqs.append(curr_seq)
        
        longest_seq = max(seqs, key=len)
        print(f"Longest valid sequence has {len(longest_seq)} markers")
        
test_row_markers('template.png')
