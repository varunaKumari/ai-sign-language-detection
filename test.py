import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# ----------- CAMERA SETUP (auto-detect backend) -----------
def open_camera():
    """Try multiple capture backends until a working one is found."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        time.sleep(0.5)
        if cap.isOpened():
            success, frame = cap.read()
            if success and frame is not None:
                print(f"✅ Camera working with backend: {backend}")
                return cap
            else:
                print(f"⚠ Camera opened but no frame captured with backend {backend}")
                cap.release()
    print("❌ No working camera backend found.")
    exit()

cap = open_camera()

# Optional smaller resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ----------- MODEL + DETECTOR -----------
detector = HandDetector(maxHands=1)
classifier = Classifier(
    r"C:\Users\sd407\Downloads\converted_keras (1)\keras_model.h5",
    r"C:\Users\sd407\Downloads\converted_keras (1)\labels.txt"
)

offset = 20
imgSize = 300
labels = ["Drink", "Eat", "Hello", "I love you", "No", "Please", "Sorry", "Thank you", "Yes"]

# ----------- MAIN LOOP -----------
while True:
    success, img = cap.read()

    if not success or img is None:
        print("⚠ Warning: No frame captured, retrying...")
        time.sleep(0.1)
        continue

    img = cv2.flip(img, 1)
    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Clamp crop coordinates
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        if y2 <= y1 or x2 <= x1:
            continue

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            continue

        aspectRatio = h / w
        try:
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

            # Prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]

            cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                          (x + 250, y - offset - 10), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (0, 255, 0), 3)

            cv2.imshow("ImageWhite", imgWhite)
        except Exception as e:
            print(f"⚠ Error: {e}")

    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()