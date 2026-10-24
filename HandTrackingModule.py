import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    frame=cv2.flip(img,1)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            cTime = time.time()
            fps = 1 / (cTime - pTime +  1e-6)
            pTime = cTime

            cv2.putText(img,str(int(fps)),(10,50), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)

            # Optional: Print each landmark with coordinates
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"ID: {id}, x: {cx}, y: {cy}")
                # Draw a small circle at a specific point (e.g., tip of index finger = id 8)
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
