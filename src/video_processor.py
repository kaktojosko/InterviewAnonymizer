import cv2
import os
import subprocess
from src.config import FACE_PADDING_PERCENT_X, FACE_PADDING_PERCENT_Y, YUNET_MIN_CONFIDENCE

class VideoProcessor:
    def __init__(self):
        # ЗАЩИТА ОТ THREAD EXPLOSION: Ограничиваем OpenCV 1 потоком на процесс.
        # Поскольку у нас много процессов (multiprocessing), многопоточность внутри OpenCV убьет CPU.
        cv2.setNumThreads(1)
        
        # Path to the YuNet ONNX model downloaded into src/models
        self.yunet_model_path = os.path.join(os.path.dirname(__file__), 'models', 'face_detection_yunet.onnx')
        self.face_detector = None
        self.current_width = 0
        self.current_height = 0
        self.scale = 1.0

    def _init_detector(self, original_width, original_height):
        # Даунскейл до 480p ТОЛЬКО для нейросети (ускорение инференса в 5 раз!)
        target_height = 480
        if original_height > target_height:
            self.scale = target_height / original_height
            self.ml_width = int(original_width * self.scale)
            self.ml_height = target_height
        else:
            self.scale = 1.0
            self.ml_width = original_width
            self.ml_height = original_height

        if self.face_detector is None or self.current_width != self.ml_width or self.current_height != self.ml_height:
            self.face_detector = cv2.FaceDetectorYN.create(
                model=self.yunet_model_path,
                config="",
                input_size=(self.ml_width, self.ml_height),
                score_threshold=YUNET_MIN_CONFIDENCE,
                nms_threshold=0.3,
                top_k=5000
            )
            self.current_width = self.ml_width
            self.current_height = self.ml_height
        else:
            self.face_detector.setInputSize((self.ml_width, self.ml_height))

    def _detect_faces(self, frame):
        """
        Detects faces using YuNet (OpenCV FaceDetectorYN).
        """
        if self.scale != 1.0:
            ml_frame = cv2.resize(frame, (self.ml_width, self.ml_height), interpolation=cv2.INTER_LINEAR)
        else:
            ml_frame = frame
            
        _, faces = self.face_detector.detect(ml_frame)
        
        detected_faces = []
        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])
                if w > 0 and h > 0:
                    if self.scale != 1.0:
                        x = int(x / self.scale)
                        y = int(y / self.scale)
                        w = int(w / self.scale)
                        h = int(h / self.scale)
                    detected_faces.append((x, y, w, h))
                
        return detected_faces

    def _match_faces(self, faces_start, faces_end, width, height):
        """
        Matches faces between two distant frames for interpolation.
        Returns a list of tuples: (start_face, end_face).
        Prevents matching to different faces by using a maximum distance threshold.
        """
        matched = []
        unmatched_end = list(faces_end)
        
        for sf in faces_start:
            if not unmatched_end:
                matched.append((sf, sf))
                continue
                
            cx1 = sf[0] + sf[2]/2
            cy1 = sf[1] + sf[3]/2
            
            best_idx = -1
            best_dist = float('inf')
            for i, ef in enumerate(unmatched_end):
                cx2 = ef[0] + ef[2]/2
                cy2 = ef[1] + ef[3]/2
                dist = (cx1 - cx2)**2 + (cy1 - cy2)**2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            # Dynamic threshold based on face size and screen width.
            max_allowed_dist = max((sf[2] * 3)**2, (width * 0.15)**2)
                    
            if best_idx != -1 and best_dist < max_allowed_dist:
                matched.append((sf, unmatched_end.pop(best_idx)))
            else:
                # Match is too far away! Treat as disappeared.
                matched.append((sf, sf))
                
        # For faces that just appeared
        for ef in unmatched_end:
            matched.append((ef, ef))
            
        return matched

    def _apply_blur(self, frame, x, y, w, h):
        """Applies a strong pixelation (mosaic) effect to the specified region with padding."""
        pad_x = int(w * FACE_PADDING_PERCENT_X)
        pad_y = int(h * FACE_PADDING_PERCENT_Y)
        
        height, width = frame.shape[:2]
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            return
            
        mosaic_size = (15, 15)
        small = cv2.resize(roi, mosaic_size, interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        
        frame[y1:y2, x1:x2] = pixelated_roi

    def process(self, input_video_path: str, output_video_path: str, fps: str = '30/1', orig_width: int = 1920, orig_height: int = 1080) -> bool:
        if not os.path.exists(input_video_path):
            return False
            
        target_width = orig_width
        target_height = orig_height
        
        # Если разрешение выше 1080p/2K (макс. сторона > 1920), делаем ресайз на лету!
        # Это сэкономит гигабайты ОЗУ и распараллелит нагрузку по всем ядрам.
        if max(orig_width, orig_height) > 1920:
            scale = 1920 / max(orig_width, orig_height)
            target_width = int(orig_width * scale)
            target_height = int(orig_height * scale)
            # Убедимся, что размеры четные (обязательно для H.264)
            target_width = target_width - (target_width % 2)
            target_height = target_height - (target_height % 2)
            
        # Initialize YuNet with the exact video dimensions
        self._init_detector(target_width, target_height)
        
        # Парсим FPS
        from fractions import Fraction
        try:
            target_fps = float(Fraction(fps))
        except:
            target_fps = 30.0
            
        # Try H.264 (avc1) first to avoid re-encoding in mux step
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (target_width, target_height))
        if not out.isOpened():
            # Fallback to mp4v if avc1 is missing in OpenCV headless
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (target_width, target_height))
        
        if not out.isOpened():
            print(f"ERROR: Could not open VideoWriter for {output_video_path}")
            return False
            
        ffmpeg_cmd_read = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-nostats',
            '-threads', '1',
            '-i', input_video_path,
            '-r', fps,
            '-vsync', '1',
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-an', '-'
        ]
        
        ffmpeg_read_process = subprocess.Popen(
            ffmpeg_cmd_read,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )
        
        from src.config import DETECT_EVERY_N_FRAMES
        import numpy as np
        
        # Пул предсозданных кадров для zero-allocation чтения!
        pool_size = DETECT_EVERY_N_FRAMES + 1
        frame_pool = [np.empty((orig_height, orig_width, 3), dtype=np.uint8) for _ in range(pool_size)]
        pool_idx = 0
        
        buffer = []
        faces_start = []
        is_first_chunk = True
        
        frame_size = orig_width * orig_height * 3
        
        try:
            while True:
                frame_array = frame_pool[pool_idx]
                frame_view = memoryview(frame_array).cast('B')
                
                bytes_read = 0
                while bytes_read < frame_size:
                    chunk = ffmpeg_read_process.stdout.readinto(frame_view[bytes_read:])
                    if not chunk:
                        break
                    bytes_read += chunk
                
                if bytes_read != frame_size:
                    if buffer:
                        last_frame = buffer[-1]
                        faces_end = self._detect_faces(last_frame)
                        if is_first_chunk:
                            faces_start = self._detect_faces(buffer[0])
                            
                        matched_pairs = self._match_faces(faces_start, faces_end, target_width, target_height)
                        
                        for i, b_frame in enumerate(buffer):
                            alpha = i / max(1, len(buffer) - 1) if len(buffer) > 1 else 0
                            for (sf, ef) in matched_pairs:
                                x = int(sf[0] * (1 - alpha) + ef[0] * alpha)
                                y = int(sf[1] * (1 - alpha) + ef[1] * alpha)
                                w = int(sf[2] * (1 - alpha) + ef[2] * alpha)
                                h = int(sf[3] * (1 - alpha) + ef[3] * alpha)
                                self._apply_blur(b_frame, x, y, w, h)
                            out.write(b_frame)
                    break
                    
                if target_width != orig_width or target_height != orig_height:
                    frame = cv2.resize(frame_array, (target_width, target_height), interpolation=cv2.INTER_AREA)
                else:
                    frame = frame_array
                    
                buffer.append(frame)
                pool_idx = (pool_idx + 1) % pool_size
                
                if len(buffer) >= DETECT_EVERY_N_FRAMES:
                    last_frame = buffer[-1]
                    faces_end = self._detect_faces(last_frame)
                    
                    if is_first_chunk:
                        faces_start = self._detect_faces(buffer[0])
                        is_first_chunk = False
                        
                    matched_pairs = self._match_faces(faces_start, faces_end, target_width, target_height)
                    
                    for i, b_frame in enumerate(buffer):
                        if is_first_chunk:
                            alpha = i / max(1, len(buffer) - 1)
                        else:
                            alpha = (i + 1) / len(buffer)
                            
                        for (sf, ef) in matched_pairs:
                            x = int(sf[0] * (1 - alpha) + ef[0] * alpha)
                            y = int(sf[1] * (1 - alpha) + ef[1] * alpha)
                            w = int(sf[2] * (1 - alpha) + ef[2] * alpha)
                            h = int(sf[3] * (1 - alpha) + ef[3] * alpha)
                            self._apply_blur(b_frame, x, y, w, h)
                        out.write(b_frame)
                        
                    is_first_chunk = False
                    
                    faces_start = faces_end
                    buffer = []
        finally:
            ffmpeg_read_process.stdout.close()
            ffmpeg_read_process.wait()
            out.release()
            
        return True
