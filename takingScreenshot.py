import cv2
import mediapipe as mp
import pyautogui
import numpy as np

vid = cv2.VideoCapture(0)
suc, bom = vid.read()
h,w,c = bom.shape


mediaPipe_hands = mp.solutions.hands
mediaPipe_drawing = mp.solutions.drawing_utils

hands = mediaPipe_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tip_ids = [4,8,12,16,20]

def countFingers(image, hand_landmarks, handNo=0):
    if hand_landmarks:
        land_mark = hand_landmarks[handNo].landmark
        fingers = [] 
        for lm in tip_ids:
            x = int(land_mark[lm].x*w)
            y = int(land_mark[lm].y*h)
            black = image
            cv2.circle(black, (x,y), 15, (255, 0, 0), cv2.FILLED)
            if lm != 4:
                finger_tip_y = land_mark[lm].y
                finger_bottom_y = land_mark[lm-2].y
                thumb_tip_x = land_mark[4].x
                thumb_bottom_x = land_mark[4-2].x
                if thumb_tip_x < thumb_bottom_x:
                    fingers.append(1)
                if thumb_tip_x > thumb_bottom_x:
                    fingers.append(0)
                if finger_tip_y < finger_bottom_y:
                    fingers.append(1)
                if finger_tip_y > finger_bottom_y:
                    fingers.append(0)
            total_fingers = fingers.count(1)
            if total_fingers == 0:
                image = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                cv2.imwrite("Screenshotss.png", image)
                   

def drawHandLanmarks(image, hand_landmarks):
    if hand_landmarks:
        for lm in hand_landmarks:
            mediaPipe_drawing.draw_landmarks(image, lm, mediaPipe_hands.HAND_CONNECTIONS)



while True:
    ret, image = vid.read()
    
    image = cv2.flip(image,1)

    results = hands.process(image)

    hand_landmarks = results.multi_hand_landmarks

    drawHandLanmarks(image, hand_landmarks)
    countFingers(image,hand_landmarks)

    cv2.imshow("ScreenShot", image)
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()