import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import modules_config as config
import os


#mediapipeを用いて各フレームの骨格座標を取得し,CSVに書き込み
def write_skeleton_csv_func(movie_path, csv_path):
    #初期設定
    cap = cv2.VideoCapture(movie_path)
    #体の設定
    base_pose_options = python.BaseOptions(model_asset_path=config.POSE_LANDMARK_MODEL_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_pose_options,
        output_segmentation_masks=True)
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
    #手の設定
    base_hands_options = python.BaseOptions(model_asset_path=config.HAND_LANDMARK_MODEL_PATH)
    hands_options = vision.HandLandmarkerOptions(base_options=base_hands_options,num_hands=2)
    hands_detector = vision.HandLandmarker.create_from_options(hands_options)
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # ヘッダーを書き込む
        headers = ['Frame']
        # pose_landmark_namesはmediapipeのposeのランドマークの名前
        pose_landmark_names = [f'pose_{i}_{axis}' for i in range(config.POSE_TOTAL_NUMBER_SKELETON) for axis in ['x', 'y', 'z']]

        left_hand_landmark_names = [f'left_hand_{i}_{axis}' for i in range(config.HAND_TOTAL_NUMBER_SKELETON) for axis in ['x', 'y', 'z']]
        right_hand_landmark_names = [f'right_hand_{i}_{axis}' for i in range(config.HAND_TOTAL_NUMBER_SKELETON) for axis in ['x', 'y', 'z']]

        headers.extend(pose_landmark_names)
        headers.extend(left_hand_landmark_names)
        headers.extend(right_hand_landmark_names)

        writer.writerow(headers)
        frame_index = 0
        while cap.isOpened():
            ret, img = cap.read()

            if not ret:
                break
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            pose_detection_result = pose_detector.detect(img)
            hands_detection_result = hands_detector.detect(img)
            row = [frame_index]
            row.extend(get_pose_landmarks(pose_detection_result))
            row.extend(get_hand_landmarks(hands_detection_result))
            writer.writerow(row)

            frame_index += 1

            # pose_annotated_image = draw_pose_landmarks_on_image(img.numpy_view(), pose_detection_result)
            # annotated_image = draw_hands_landmarks_on_image(pose_annotated_image, hands_detection_result)

            # cv2.imshow("img",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cv2.destroyAllWindows()

def get_pose_landmarks(pose_detection_result):
    if not pose_detection_result.pose_landmarks:
        return [None] * config.POSE_TOTAL_NUMBER_SKELETON * config.DIMENSION
    landmarks = []
    for landmark in pose_detection_result.pose_landmarks[0]:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    return landmarks

def get_hand_landmarks(detection_result):
    landmarks = []
    left_hand_landmarks = [None] * config.HAND_TOTAL_NUMBER_SKELETON * config.DIMENSION
    right_hand_landmarks = [None] * config.HAND_TOTAL_NUMBER_SKELETON * config.DIMENSION
    if len(detection_result.hand_landmarks)==2:
        if detection_result.handedness[0][0].category_name == detection_result.handedness[1][0].category_name: #同じ手の時
            return left_hand_landmarks + right_hand_landmarks
        else:
              for hand_landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):
                  hand_label = handedness[0].category_name
                  landmarks = []
                  for landmark in hand_landmarks:
                      landmarks.extend([landmark.x, landmark.y  , landmark.z])
                  if hand_label == "Left":
                      left_hand_landmarks = landmarks
                  else:
                      right_hand_landmarks = landmarks
              return left_hand_landmarks + right_hand_landmarks
    elif len(detection_result.hand_landmarks)==1:
        hand = detection_result.handedness[0][0].category_name
        if hand == "Left":
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            left_hand_landmarks = landmarks
        else:
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            right_hand_landmarks = landmarks
        landmarks = left_hand_landmarks + right_hand_landmarks
    else:
       landmarks = left_hand_landmarks + right_hand_landmarks
    return landmarks

def draw_hands_landmarks_on_image(image, detection_result):
    annotated_image = image.copy()

    if len(detection_result.hand_landmarks)==2:
        if detection_result.handedness[0][0].category_name == detection_result.handedness[1][0].category_name:
            return annotated_image
    for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
        hands = detection_result.handedness[i][0].category_name

        if hands == "Left":
           color = (0, 0, 0)
        else:
           color = (255, 255, 255)
        for landmark in hand_landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(annotated_image, (x, y), 5, color, -1)
    return annotated_image

def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          pose_landmarks_proto,
          solutions.pose.POSE_CONNECTIONS,
          solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


if __name__ == "__main__":
  folder_path = "../../LSA64/all/"
  files_and_directories = os.listdir(folder_path)
  file_names = [f for f in files_and_directories if os.path.isfile(os.path.join(folder_path, f))]
  sorted_file_names = sorted(file_names)
  for i, file_name in enumerate(sorted_file_names):
      file_path = folder_path + file_name
      store_csv_file_path = "../../csv/LSA64/" + file_name[:-4] + ".csv"
      print(file_path, store_csv_file_path)
      write_skeleton_csv_func(file_path, store_csv_file_path)