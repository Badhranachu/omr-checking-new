import cv2
import numpy as np
import json

img = cv2.imread('template.png')
# Assuming template.png is perfectly aligned or we can just resize it to our standard size directly
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
warped = cv2.resize(gray, (1000, 1600))

# Try Otsu to find black shapes
_, thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

bubbles = []
for c in contours:
    area = cv2.contourArea(c)
    # A standard bubble at 1000x1600 usually has an area
    if 30 < area < 1000:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if 0.6 < ar < 1.4:
            # Check solidity to ensure it's circular/oval
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and (area / hull_area) > 0.6:
                cx = x + w // 2
                cy = y + h // 2
                bubbles.append((cx, cy, max(w, h) // 2))

print(f"Total circle-like shapes found: {len(bubbles)}")

# Let's group them by rows (Y)
rows = {}
for cx, cy, r in bubbles:
    found = False
    for ry in rows.keys():
        if abs(ry - cy) < 10:
            rows[ry].append((cx, cy, r))
            found = True
            break
    if not found:
        rows[cy] = [(cx, cy, r)]

# A valid question row would typically have at least 4 options (A,B,C,D)
valid_rows = {k: v for k, v in rows.items() if len(v) >= 2}
print(f"Valid rows (>=2 bubbles) found: {len(valid_rows)}")

# If we found 200 rows with bubbles, we did it!
