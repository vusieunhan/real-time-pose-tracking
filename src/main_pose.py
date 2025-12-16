# src/main_pose.py

import cv2
from pose import PoseEstimator

def draw_keypoints(frame, keypoints):
    for person in keypoints:
        for x, y in person:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

def main():
    cap = cv2.VideoCapture("data\\test_walk.mp4")
    pose_estimator = PoseEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = pose_estimator.estimate(frame)

        if keypoints is not None:
            draw_keypoints(frame, keypoints)

        cv2.imshow("Pose", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
