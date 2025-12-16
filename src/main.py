# src/main.py
import torch
import cv2
import time
from detector import PersonDetector
from tracker import PersonTracker

def main():
    cap = cv2.VideoCapture("test1.mp4")

    detector = PersonDetector(
        model_path="yolov8n.pt",
        device="cuda"
    )

    tracker = PersonTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        
        with torch.no_grad():
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)

        fps = 1 / (time.time() - start)

        for track in tracks:
            
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            cv2.rectangle(
                frame,
                (l, t),
                (r, b),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"ID {track_id}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Day 3 - Tracking", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
