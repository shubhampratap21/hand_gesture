import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Initialize Pygame mixer
pygame.mixer.init()

# Playlist configuration
playlist = [
    r"D:\Projects\hand gesture\playlist\song1.mp3",
    r"D:\Projects\hand gesture\playlist\song2.mp3",
    r"D:\Projects\hand gesture\playlist\song3.mp3"
]
current_song_index = 0
pygame.mixer.music.load(playlist[current_song_index])
pygame.mixer.music.play()

# Global variables
volume = 1.0
cooldown = time.time()

def is_thumbs_up(hand_landmarks, frame_width, frame_height):
    # Check if thumb is extended more than other fingers
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    return thumb_tip.y < index_tip.y

def get_distance(landmark1, landmark2, frame_width, frame_height):
    x1, y1 = int(landmark1.x * frame_width), int(landmark1.y * frame_height)
    x2, y2 = int(landmark2.x * frame_width), int(landmark2.y * frame_height)
    return math.hypot(x2 - x1, y2 - y1)

# Main loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    frame_height, frame_width, _ = frame.shape
    hands_detected = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hands_detected.append((hand_landmarks, handedness.classification[0].label))

    # Gesture detection
    if len(hands_detected) > 0:
        # Volume Control (Single hand - thumb to pinky distance)
        if len(hands_detected) == 1:
            hand, label = hands_detected[0]
            thumb_tip = hand.landmark[4]
            pinky_tip = hand.landmark[20]
            
            distance = get_distance(thumb_tip, pinky_tip, frame_width, frame_height)
            volume = np.interp(distance, [50, 250], [0.0, 1.0])
            pygame.mixer.music.set_volume(volume)
            
            cv2.putText(frame, f"Volume: {int(volume*100)}%", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Play/Pause (Two hands - index fingers touching)
        if len(hands_detected) == 2 and time.time() > cooldown:
            hand1, hand2 = hands_detected[0][0], hands_detected[1][0]
            index1 = hand1.landmark[8]
            index2 = hand2.landmark[8]
            
            if get_distance(index1, index2, frame_width, frame_height) < 30:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.pause()
                else:
                    pygame.mixer.music.unpause()
                cooldown = time.time() + 3

        # Skip Song (Right hand thumbs up)
        for hand, label in hands_detected:
            if label == "Right" and is_thumbs_up(hand, frame_width, frame_height):
                if time.time() > cooldown:
                    current_song_index = (current_song_index + 1) % len(playlist)
                    pygame.mixer.music.load(playlist[current_song_index])
                    pygame.mixer.music.play()
                    cooldown = time.time() + 3

        # Rewind Song (Left hand thumbs up)
        for hand, label in hands_detected:
            if label == "Left" and is_thumbs_up(hand, frame_width, frame_height):
                if time.time() > cooldown:
                    pygame.mixer.music.rewind()
                    cooldown = time.time() + 3

    # Display current song
    cv2.putText(frame, f"Now Playing: {playlist[current_song_index].split('\\')[-1]}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Gesture Music Controller', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()