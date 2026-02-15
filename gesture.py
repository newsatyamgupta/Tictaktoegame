import cv2
import mediapipe as mp
import pyautogui
import math
import time
 

pyautogui.FAILSAFE = False

# -----------------------------
# Mediapipe hands setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# -----------------------------
# Camera setup
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Screen size
screen_w, screen_h = pyautogui.size()

# -----------------------------
# Function to calculate distance
# -----------------------------

# Distance Formula
# Distance=(x2−x1)2+(y2−y1)2
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not captured!")
        break

    # Mirror view
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Landmarks
        index_tip = hand.landmark[8]
        thumb_tip = hand.landmark[4]
        middle_tip = hand.landmark[12]

        # Smooth cursor movement
        curr_x, curr_y = pyautogui.position()
        x = int(curr_x + (index_tip.x * screen_w - curr_x) * 0.3)
        y = int(curr_y + (index_tip.y * screen_h - curr_y) * 0.3)
        pyautogui.moveTo(x, y)

        # Click (thumb + index)
        if distance(index_tip, thumb_tip) < 0.05:
            pyautogui.click()
            cv2.putText(frame, "CLICK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            time.sleep(0.2)

        # Scroll (index + middle)
        if distance(index_tip, middle_tip) < 0.05:
            pyautogui.scroll(-20)
            cv2.putText(frame, "SCROLL", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            time.sleep(0.2)

    else:
        # Hand not detected message
        cv2.putText(frame, "Show hand to control", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Show frame
    cv2.imshow("Gesture Control", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release
cap.release()
cv2.destroyAllWindows()