#Sauce: https://natakaro.gumroad.com/l/oprmi
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

standardCon = {(15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8), (15, 17),
               (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4),
               (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), (2, 3),
               (11, 12), (27, 29), (13, 15)}

customCon = {(3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8), (24, 26), (4, 5), (5, 6), (29, 31),
             (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (25, 27), (26, 28),
             (12, 14), (2, 3), (11, 12), (27, 29), (13, 15)}

right_eye = {(0, 4), (4, 5), (5, 6), (6, 8)}
left_eye = {(0, 1), (1, 2), (2, 3), (3, 7)}
right_hand = {(14, 16), (12, 14)}
left_hand = {(11, 13), (13, 15)}
right_leg = {(24, 26), (26, 28)}
left_leg = {(23, 25), (25, 27)}
right_foot = {(28, 32), (28, 30), (30, 32)}
left_foot = {(27, 29), (27, 31), (29, 31)}
right_torso = (12, 24)
left_torso = (11, 23)

right_eye_color = (255, 0, 255)
left_eye_color = (0, 0, 255)
right_hand_color = (255, 40, 255)
left_hand_color = (40, 40, 255)
right_torso_color = (255, 80, 255)
left_torso_color = (80, 80, 255)
right_leg_color = (255, 120, 255)
left_leg_color = (120, 120, 255)
right_foot_color = (255, 160, 255)
left_foot_color = (160, 160, 255)

connection_annotations = {}
for connection in standardCon:
    connection_annotations[connection] = mp_drawing.DrawingSpec()

for connection in right_eye:
    connection_annotations[connection] = mp_drawing.DrawingSpec(right_eye_color)

for connection in left_eye:
    connection_annotations[connection] = mp_drawing.DrawingSpec(left_eye_color)

for connection in right_hand:
    connection_annotations[connection] = mp_drawing.DrawingSpec(right_hand_color)

for connection in left_hand:
    connection_annotations[connection] = mp_drawing.DrawingSpec(left_hand_color)

for connection in right_leg:
    connection_annotations[connection] = mp_drawing.DrawingSpec(right_leg_color)

for connection in left_leg:
    connection_annotations[connection] = mp_drawing.DrawingSpec(left_leg_color)

for connection in right_foot:
    connection_annotations[connection] = mp_drawing.DrawingSpec(right_foot_color)

for connection in left_foot:
    connection_annotations[connection] = mp_drawing.DrawingSpec(left_foot_color)

connection_annotations[right_torso] = mp_drawing.DrawingSpec(right_torso_color)
connection_annotations[left_torso] = mp_drawing.DrawingSpec(left_torso_color)

def apply_mediapipe(input_image, detect_pose = True, detect_hands = True):
    image_height, image_width, _ = input_image.shape
    annotated_image = np.zeros((image_height, image_width, 3), np.uint8)

    pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5)

    hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=8,
            min_detection_confidence=0.5)

    if detect_pose:
        resultsPose = pose.process(input_image)
        if resultsPose.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                resultsPose.pose_landmarks,
                customCon,
                landmark_drawing_spec=None,
                connection_drawing_spec=connection_annotations)
    if detect_hands:
        resultsHands = hands.process(input_image)
        if resultsHands.multi_hand_landmarks:
            for hand_landmarks in resultsHands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    None,
                    mp_drawing_styles.get_default_hand_connections_style())
    return annotated_image
