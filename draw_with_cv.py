import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw = mp.solutions.drawing_utils

drawing = None

x3, y3 = None, None


x1, y1 = None, None
x2, y2 = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 
    h, w, _ = frame.shape
    
    if drawing is None:
        drawing = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        drawing[:] = 0  

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    op = hands.process(rgb)
    
    if op.multi_hand_landmarks:
        for hand_landmarks in op.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            for id, landmark in enumerate(landmarks):
                X = int(landmark.x * w)
                Y = int(landmark.y * h)
                if id == 8:  
                    x1 = X
                    y1 = Y
                    cv2.circle(frame, (X, Y), 8, (0, 0, 255), 3)
                if id == 4:  
                    cv2.circle(frame, (X, Y), 8, (0, 255, 0), 3)
                    x2 = X
                    y2 = Y

            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if dist < 30:  
                    if x3 is not None and y3 is not None:
                        cv2.line(drawing, (x3, y3), (x1, y1), color=(255, 255, 255), thickness=3)
                x3, y3 = x1, y1    

    else:
        x3, y3 = None, None  

    
    frame = cv2.addWeighted(frame, 0.5, cv2.cvtColor(drawing, cv2.COLOR_GRAY2BGR), 0.5, 0)
    
    cv2.imshow("video", frame)
    
    if cv2.waitKey(1) == ord("x"):
        break


cap.release()
cv2.destroyAllWindows()
