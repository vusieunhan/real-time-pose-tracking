# src/behavior.py

import numpy as np
from collections import deque

class PoseBehaviorAnalyzer:
    def __init__(self, history_len=5):
        self.history_len = history_len
        self.hip_history = {}  # track_id -> deque

    def update(self, track_id, keypoints):
        """
        keypoints: (17, 2)
        """
        LEFT_HIP = 11
        RIGHT_HIP = 12

        hip_y = np.mean([
            keypoints[LEFT_HIP][1],
            keypoints[RIGHT_HIP][1]
        ])

        if track_id not in self.hip_history:
            self.hip_history[track_id] = deque(maxlen=self.history_len)

        self.hip_history[track_id].append(hip_y)

    def classify(self, track_id):
        if track_id not in self.hip_history:
            return "Unknown"

        hist = self.hip_history[track_id]
        if len(hist) < self.history_len:
            return "Standing"

        velocity = np.mean(np.abs(np.diff(hist)))

        if velocity < 2:
            return "Standing"
        elif velocity < 6:
            return "Walking"
        else:
            return "Running"
