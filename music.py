import cv2
import numpy as np
import mediapipe as mp
import pygame

# Initialize Pygame for sound
pygame.init()

# Sound files (make sure paths are correct)
key_sounds = {
    0: pygame.mixer.Sound("sounds/G.wav"),
    1: pygame.mixer.Sound("sounds/C_high.wav"),
    2: pygame.mixer.Sound("sounds/C.wav"),
    3: pygame.mixer.Sound("sounds/B.wav"),
    4: pygame.mixer.Sound("sounds/D.wav"),
    5: pygame.mixer.Sound("sounds/E.wav"),
    6: pygame.mixer.Sound("sounds/F.wav"),
    7: pygame.mixer.Sound("sounds/A.wav"),
}

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Setup camera
cap = cv2.VideoCapture(0)
WIDTH, HEIGHT = 800, 600
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Define piano key regions (8 keys)
key_regions = [(i * (WIDTH // 8), (i + 1) * (WIDTH // 8)) for i in range(8)]

# Track last played key to prevent repeated sound
last_played_key = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Flip horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw piano keys (colored blocks)
    for i, (x_start, x_end) in enumerate(key_regions):
        cv2.rectangle(frame, (x_start, HEIGHT - 100), (x_end, HEIGHT), (255, 255, 255), -1)
        cv2.putText(frame, f"Key {i}", (x_start + 10, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Process hand landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get fingertip position (Index Finger Tip)
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip
            x, y = int(index_finger_tip.x * WIDTH), int(index_finger_tip.y * HEIGHT)

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Check if the index finger touches any key
            for i, (x_start, x_end) in enumerate(key_regions):
                if x_start < x < x_end and (HEIGHT - 100) < y < HEIGHT:
                    if last_played_key != i:
                        key_sounds[i].play()
                        last_played_key = i
                    break
            else:
                last_played_key = None  # Reset if finger is not on a key

    cv2.imshow("Virtual Piano", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# if you want to run this code fristly you  need some libararry  like = numpy, pygame, mediapipe, and make sure your puythone verison is 3.11 bcoz the open cv properly work in this livary 
#sounds file is not correct adderess so i will be provide addersss sounds wav file 
