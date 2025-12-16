# src/main_pose_bbox.py

import cv2
from detector import PersonDetector
from pose import PoseEstimator

def draw_pose(frame, keypoints, offset_x, offset_y):
    for x, y in keypoints:
        cv2.circle(
            frame,
            (int(x + offset_x), int(y + offset_y)),
            3,
            (0, 255, 0),
            -1
        )

def main():
    cap = cv2.VideoCapture("test.mp4")

    detector = PersonDetector(
        model_path="yolov8n.pt",
        device="cuda"
    )
    pose_estimator = PoseEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for bbox, score, _ in detections:
            x, y, w, h = map(int, bbox)
            person_crop = frame[y:y+h, x:x+w]

            if person_crop.size == 0:
                continue

            keypoints = pose_estimator.estimate(person_crop)
            if keypoints is None or len(keypoints) == 0:
                continue

            draw_pose(frame, keypoints[0], x, y)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

        cv2.imshow("Week 2 Day 2 - Pose by BBox", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
