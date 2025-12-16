# src/fall_detector.py

import numpy as np
from collections import deque

class FallDetector:
    def __init__(self, history_len=6):
        self.history_len = history_len
        self.hip_y_history = {}

    def update(self, track_id, keypoints):
        LEFT_HIP = 11
        RIGHT_HIP = 12

        hip_y = np.mean([
            keypoints[LEFT_HIP][1],
            keypoints[RIGHT_HIP][1]
        ])

        if track_id not in self.hip_y_history:
            self.hip_y_history[track_id] = deque(maxlen=self.history_len)

        self.hip_y_history[track_id].append(hip_y)

    def detect(self, track_id):
        if track_id not in self.hip_y_history:
            return False

        hist = self.hip_y_history[track_id]
        if len(hist) < self.history_len:
            return False

        delta = hist[-1] - hist[0]

        # té ngã: hip tụt nhanh
        if delta > 40:
            return True

        return False
