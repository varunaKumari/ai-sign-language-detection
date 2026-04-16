import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

# Folder to save images
folder = r"C:\Users\sd407\OneDrive\Desktop\Sign Language Detection\data\yes"
os.makedirs(folder, exist_ok=True)  # ✅ create folder automatically if missing

while True:
    success, img = cap.read()
    if not success:
        print("❌ Camera not detected.")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # ✅ Make sure crop coordinates are inside the image
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        # ✅ Skip empty crops (prevents resize crash)
        if imgCrop.size == 0:
            print("⚠️ Skipping empty crop (hand near edge)")
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Show windows
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        image_path = f"{folder}/Image_{time.time()}.jpg"
        cv2.imwrite(image_path, imgWhite)
        print(f"✅ Saved: {image_path} ({counter})")

    elif key == ord("q"):  # Press q to quit
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
