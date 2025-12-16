# src/main_behavior.py

import cv2
import torch
from detector import PersonDetector
from tracker import PersonTracker
from pose import PoseEstimator
from behavior import PoseBehaviorAnalyzer

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
    cap = cv2.VideoCapture("data/test_walk.mp4")

    detector = PersonDetector("yolov8n.pt", "cuda")
    tracker = PersonTracker()
    pose_estimator = PoseEstimator()
    behavior_analyzer = PoseBehaviorAnalyzer()

    frame_count = 0
    POSE_EVERY_N = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        with torch.no_grad():
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            if frame_count % POSE_EVERY_N != 0:
                continue

            person_crop = frame[t:b, l:r]
            if person_crop.size == 0:
                continue

            keypoints = pose_estimator.estimate(person_crop)
            if keypoints is None or len(keypoints) == 0:
                continue

            pose = keypoints[0]
            behavior_analyzer.update(track_id, pose)
            behavior = behavior_analyzer.classify(track_id)

            draw_pose(frame, pose, l, t)

            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id} | {behavior}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        cv2.imshow("Behavior Analysis", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
