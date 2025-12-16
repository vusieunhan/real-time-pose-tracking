# src/main_fall.py

import cv2
import torch
from detector import PersonDetector
from tracker import PersonTracker
from pose import PoseEstimator
from fall_detector import FallDetector

torch.backends.cudnn.benchmark = True

def main():
    cap = cv2.VideoCapture("data\\test_walk.mp4")

    detector = PersonDetector("yolov8n.pt", "cuda")
    tracker = PersonTracker()
    pose_estimator = PoseEstimator()
    fall_detector = FallDetector()

    frame_count = 0
    POSE_EVERY_N = 8   # pose mỗi 4 frame → tăng FPS

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
            if track.time_since_update > 0:
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
            fall_detector.update(track_id, pose)

            if fall_detector.detect(track_id):
                cv2.putText(
                    frame,
                    "FALL DETECTED",
                    (l, t - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)

        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
