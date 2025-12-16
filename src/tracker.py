# src/tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=60,              # giữ track lâu hơn
            n_init=2,                # confirm nhanh
            max_iou_distance=0.95,   # ưu tiên IOU mạnh
            half=True                # FP16
        )

    def update(self, detections, frame):
        return self.tracker.update_tracks(
            detections,
            frame=frame
        )
