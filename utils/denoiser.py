import cv2
import numpy as np
import time

class AdaptiveDenoiser:
    """
    A lightweight, frame-adaptive denoising module specifically tuned for YOLO.
    It estimates noise levels and applies edge-preserving filtering only when necessary.
    """
    def __init__(self, noise_threshold=15.0):
        self.noise_threshold = noise_threshold
        self.latency_log = []

    def estimate_noise(self, frame):
        """
        Estimate noise using the standard deviation of the Laplacian on a downsampled image.
        Downsampling makes it much faster (lightweight).
        """
        # Downsample for faster estimation
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (w // 2, h // 2))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate noise level
        # We use a 3x3 Laplacian kernel
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = np.std(laplacian)
        return noise_level

    def process_frame(self, frame):
        """
        Main pipeline: Estimate -> Decide -> Denoise (if needed).
        """
        start_time = time.time()
        
        noise_level = self.estimate_noise(frame)
        
        # Thresholds tuned for YOLO (preserving edges while removing high-freq noise)
        if noise_level < self.noise_threshold:
            # Low noise: Skip denoising to minimize latency
            result = frame
            denoised = False
        else:
            # Medium to High noise: Apply lightweight edge-preserving filter
            # Bilateral filter is used as it preserves edges crucial for YOLO detection
            # Parameters (d=5, sigmaColor=50, sigmaSpace=50) provide a good balance
            result = cv2.bilateralFilter(frame, 5, 50, 50)
            denoised = True
            
        latency = (time.time() - start_time) * 1000 # in ms
        self.latency_log.append(latency)
        
        return result, noise_level, denoised, latency

    def get_average_latency(self):
        if not self.latency_log:
            return 0
        return sum(self.latency_log) / len(self.latency_log)
