import threading
import time
import numpy as np

class FrameManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        # Two separate frames with their own locks
        self._frame1 = None
        self._frame2 = None
        self._frame1_lock = threading.Lock()
        self._frame2_lock = threading.Lock()
        self._frame1_time = 0
        self._frame2_time = 0
        self._frame1_count = 0
        self._frame2_count = 0
    
    def update_frame1(self, new_frame):
        """Update frame 1 (called by external app)"""
        with self._frame1_lock:
            if new_frame is not None:
                self._frame1 = new_frame.copy()
                self._frame1_time = time.time()
                self._frame1_count += 1
    
    def update_frame2(self, new_frame):
        """Update frame 2 (called by external app)"""
        with self._frame2_lock:
            if new_frame is not None:
                self._frame2 = new_frame.copy()
                self._frame2_time = time.time()
                self._frame2_count += 1
    
    def update_both_frames(self, frame1, frame2):
        """Update both frames atomically"""
        with self._frame1_lock:
            with self._frame2_lock:
                if frame1 is not None:
                    self._frame1 = frame1.copy()
                    self._frame1_time = time.time()
                    self._frame1_count += 1
                if frame2 is not None:
                    self._frame2 = frame2.copy()
                    self._frame2_time = time.time()
                    self._frame2_count += 1
    
    def get_frame1(self):
        """Get frame 1 (called by stream server)"""
        with self._frame1_lock:
            if self._frame1 is not None:
                return self._frame1.copy(), self._frame1_time, self._frame1_count
        return None, 0, 0
    
    def get_frame2(self):
        """Get frame 2 (called by stream server)"""
        with self._frame2_lock:
            if self._frame2 is not None:
                return self._frame2.copy(), self._frame2_time, self._frame2_count
        return None, 0, 0
    
    def get_frame_info(self):
        """Get info for both frames"""
        with self._frame1_lock:
            with self._frame2_lock:
                return {
                    'frame1': {
                        'has_frame': self._frame1 is not None,
                        'frame_time': self._frame1_time,
                        'frame_count': self._frame1_count,
                        'frame_shape': self._frame1.shape if self._frame1 is not None else None
                    },
                    'frame2': {
                        'has_frame': self._frame2 is not None,
                        'frame_time': self._frame2_time,
                        'frame_count': self._frame2_count,
                        'frame_shape': self._frame2.shape if self._frame2 is not None else None
                    }
                }