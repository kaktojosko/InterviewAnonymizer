import cv2
import numpy as np
import os
import subprocess
from fractions import Fraction
from src.config import (
    DETECT_EVERY_N_FRAMES,
    FACE_PADDING_PERCENT_X,
    FACE_PADDING_PERCENT_Y,
    OUTPUT_VIDEO_CRF,
    OUTPUT_VIDEO_PRESET,
    YUNET_MIN_CONFIDENCE,
)


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
        self.frame_width = 0
        self.frame_height = 0
        self.scale = 1.0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self._ml_yuv_1d = None
        self._ml_yuv_2d = None
        self._ml_y = None
        self._ml_u = None
        self._ml_v = None
        self._ml_bgr = None
        self._small_y = np.empty((15, 15), dtype=np.uint8)
        self._small_uv = np.empty((15, 15), dtype=np.uint8)
        self._small_bgr = np.empty((15, 15, 3), dtype=np.uint8)

    def _init_detector(self, original_width: int, original_height: int):
        self.frame_width = original_width
        self.frame_height = original_height

        # Даунскейл до 480p ТОЛЬКО для нейросети (ускорение инференса в 5 раз!)
        target_height = 480
        if original_height > target_height:
            raw_scale = target_height / original_height
            self.ml_width = max(2, int(original_width * raw_scale))
            self.ml_width = max(2, self.ml_width - (self.ml_width % 2))
            self.ml_height = target_height
        else:
            self.ml_width = max(2, original_width - (original_width % 2))
            self.ml_height = max(2, original_height - (original_height % 2))

        self.scale_x = self.ml_width / max(1, original_width)
        self.scale_y = self.ml_height / max(1, original_height)
        self.scale = self.scale_y

        if (
            self.face_detector is None
            or self.current_width != self.ml_width
            or self.current_height != self.ml_height
        ):
            self.face_detector = cv2.FaceDetectorYN.create(
                model=self.yunet_model_path,
                config="",
                input_size=(self.ml_width, self.ml_height),
                score_threshold=YUNET_MIN_CONFIDENCE,
                nms_threshold=0.3,
                top_k=5000,
            )
            self.current_width = self.ml_width
            self.current_height = self.ml_height

            # Preallocate 480p I420 + BGR buffers so _detect_faces does zero heap allocations
            ml_y_size = self.ml_width * self.ml_height
            ml_uv_size = (self.ml_width >> 1) * (self.ml_height >> 1)
            self._ml_yuv_1d = np.empty(ml_y_size + 2 * ml_uv_size, dtype=np.uint8)
            self._ml_y = self._ml_yuv_1d[:ml_y_size].reshape((self.ml_height, self.ml_width))
            self._ml_u = self._ml_yuv_1d[ml_y_size : ml_y_size + ml_uv_size].reshape(
                (self.ml_height >> 1, self.ml_width >> 1)
            )
            self._ml_v = self._ml_yuv_1d[ml_y_size + ml_uv_size :].reshape(
                (self.ml_height >> 1, self.ml_width >> 1)
            )
            self._ml_yuv_2d = self._ml_yuv_1d.reshape((self.ml_height * 3 // 2, self.ml_width))
            self._ml_bgr = np.empty((self.ml_height, self.ml_width, 3), dtype=np.uint8)
        else:
            self.face_detector.setInputSize((self.ml_width, self.ml_height))

    def _detect_faces(self, frame):
        """
        Detects faces using YuNet (OpenCV FaceDetectorYN).
        Supports planar YUV420p (1-D uint8 buffer) with 480p-only color conversion,
        as well as standard 3-D BGR frames.
        """
        if frame.ndim == 1:
            width = self.frame_width
            height = self.frame_height
            if width == self.ml_width and height == self.ml_height:
                yuv_2d = frame.reshape((height * 3 // 2, width))
                ml_frame = cv2.cvtColor(yuv_2d, cv2.COLOR_YUV2BGR_I420, dst=self._ml_bgr)
            else:
                y_size = width * height
                uv_size = (width >> 1) * (height >> 1)
                y_plane = frame[:y_size].reshape((height, width))
                u_plane = frame[y_size : y_size + uv_size].reshape((height >> 1, width >> 1))
                v_plane = frame[y_size + uv_size :].reshape((height >> 1, width >> 1))
                cv2.resize(
                    y_plane, (self.ml_width, self.ml_height), dst=self._ml_y, interpolation=cv2.INTER_LINEAR
                )
                cv2.resize(
                    u_plane,
                    (self.ml_width >> 1, self.ml_height >> 1),
                    dst=self._ml_u,
                    interpolation=cv2.INTER_LINEAR,
                )
                cv2.resize(
                    v_plane,
                    (self.ml_width >> 1, self.ml_height >> 1),
                    dst=self._ml_v,
                    interpolation=cv2.INTER_LINEAR,
                )
                ml_frame = cv2.cvtColor(self._ml_yuv_2d, cv2.COLOR_YUV2BGR_I420, dst=self._ml_bgr)
        else:
            if frame.shape[1] != self.ml_width or frame.shape[0] != self.ml_height:
                ml_frame = cv2.resize(
                    frame, (self.ml_width, self.ml_height), dst=self._ml_bgr, interpolation=cv2.INTER_LINEAR
                )
            else:
                ml_frame = frame

        _, faces = self.face_detector.detect(ml_frame)

        detected_faces = []
        if faces is not None:
            inv_sx = 1.0 / self.scale_x
            inv_sy = 1.0 / self.scale_y
            need_rescale = (self.scale_x != 1.0) or (self.scale_y != 1.0)
            for face in faces:
                x, y, w, h = map(int, face[:4])
                if w > 0 and h > 0:
                    if need_rescale:
                        x = int(x * inv_sx)
                        y = int(y * inv_sy)
                        w = int(w * inv_sx)
                        h = int(h * inv_sy)
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

            cx1 = sf[0] + sf[2] / 2
            cy1 = sf[1] + sf[3] / 2

            best_idx = -1
            best_dist = float('inf')
            for i, ef in enumerate(unmatched_end):
                cx2 = ef[0] + ef[2] / 2
                cy2 = ef[1] + ef[3] / 2
                dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            # Dynamic threshold based on face size and screen width.
            max_allowed_dist = max((sf[2] * 3) ** 2, (width * 0.15) ** 2)

            if best_idx != -1 and best_dist < max_allowed_dist:
                matched.append((sf, unmatched_end.pop(best_idx)))
            else:
                # Match is too far away! Treat as disappeared.
                matched.append((sf, sf))

        # For faces that just appeared
        for ef in unmatched_end:
            matched.append((ef, ef))

        return matched

    def _apply_blur(self, frame, x, y, w, h, frame_width: int = 0, frame_height: int = 0):
        """Applies a strong pixelation (mosaic) effect in-place on planar YUV420p or BGR frames."""
        pad_x = int(w * FACE_PADDING_PERCENT_X)
        pad_y = int(h * FACE_PADDING_PERCENT_Y)

        if frame.ndim == 1:
            width = frame_width or self.frame_width
            height = frame_height or self.frame_height
            # Align ROI to even coordinates so Y and 2x2 subsampled U/V planes match cleanly
            x1 = max(0, (x - pad_x) & ~1)
            y1 = max(0, (y - pad_y) & ~1)
            x2 = min(width, (x + w + pad_x + 1) & ~1)
            y2 = min(height, (y + h + pad_y + 1) & ~1)
            bw = x2 - x1
            bh = y2 - y1
            if bw < 2 or bh < 2:
                return

            y_size = width * height
            uv_size = (width >> 1) * (height >> 1)
            y_plane = frame[:y_size].reshape((height, width))
            u_plane = frame[y_size : y_size + uv_size].reshape((height >> 1, width >> 1))
            v_plane = frame[y_size + uv_size :].reshape((height >> 1, width >> 1))

            y_roi = y_plane[y1:y2, x1:x2]
            cv2.resize(y_roi, (15, 15), dst=self._small_y, interpolation=cv2.INTER_LINEAR)
            out_y = cv2.resize(self._small_y, (bw, bh), dst=y_roi, interpolation=cv2.INTER_NEAREST)
            if out_y is not y_roi:
                y_plane[y1:y2, x1:x2] = out_y

            ux1, uy1, ux2, uy2 = x1 >> 1, y1 >> 1, x2 >> 1, y2 >> 1
            ubw, ubh = ux2 - ux1, uy2 - uy1

            u_roi = u_plane[uy1:uy2, ux1:ux2]
            cv2.resize(u_roi, (15, 15), dst=self._small_uv, interpolation=cv2.INTER_LINEAR)
            out_u = cv2.resize(self._small_uv, (ubw, ubh), dst=u_roi, interpolation=cv2.INTER_NEAREST)
            if out_u is not u_roi:
                u_plane[uy1:uy2, ux1:ux2] = out_u

            v_roi = v_plane[uy1:uy2, ux1:ux2]
            cv2.resize(v_roi, (15, 15), dst=self._small_uv, interpolation=cv2.INTER_LINEAR)
            out_v = cv2.resize(self._small_uv, (ubw, ubh), dst=v_roi, interpolation=cv2.INTER_NEAREST)
            if out_v is not v_roi:
                v_plane[uy1:uy2, ux1:ux2] = out_v
            return

        height, width = frame.shape[:2]
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            return

        cv2.resize(roi, (15, 15), dst=self._small_bgr, interpolation=cv2.INTER_LINEAR)
        out_bgr = cv2.resize(
            self._small_bgr, (x2 - x1, y2 - y1), dst=roi, interpolation=cv2.INTER_NEAREST
        )
        if out_bgr is not roi:
            frame[y1:y2, x1:x2] = out_bgr

    def process(
        self,
        input_video_path: str,
        output_video_path: str,
        fps: str = '30/1',
        orig_width: int = 1920,
        orig_height: int = 1080,
        enc_threads: int = 2,
        cancel_event=None,
    ) -> bool:
        if cancel_event is not None and cancel_event.is_set():
            return False
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

        # Четные размеры обязательны для YUV420p и H.264 (даже если исходник был с нечетной стороной)
        target_width = max(2, target_width - (target_width % 2))
        target_height = max(2, target_height - (target_height % 2))

        # Initialize YuNet with the exact video dimensions
        self._init_detector(target_width, target_height)

        # Парсим FPS
        try:
            target_fps = float(Fraction(fps))
        except Exception:
            target_fps = 30.0

        enc_threads_str = str(max(1, int(enc_threads)))

        # Инициализируем FFmpeg для прямой записи YUV420p -> H.264 без конвертации цвета!
        # Важно: -threads ставится ПОСЛЕ -i - (для libx264), чтобы воркеры не плодили по 20+ потоков.
        ffmpeg_cmd_write = [
            'ffmpeg',
            '-y',
            '-hide_banner',
            '-loglevel', 'error',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f"{target_width}x{target_height}",
            '-pix_fmt', 'yuv420p',
            '-r', str(target_fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-threads', enc_threads_str,
            '-preset', OUTPUT_VIDEO_PRESET,
            '-crf', str(OUTPUT_VIDEO_CRF),
            '-pix_fmt', 'yuv420p',
            output_video_path,
        ]

        try:
            write_process = subprocess.Popen(
                ffmpeg_cmd_write,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=8 * 1024 * 1024,
            )
        except Exception as e:
            print(f"ERROR: Could not open FFmpeg write pipe for {output_video_path}: {e}")
            return False

        ffmpeg_cmd_read = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-nostats',
            '-threads', '1',
            '-i', input_video_path,
        ]
        if target_width != orig_width or target_height != orig_height:
            ffmpeg_cmd_read.extend(['-s', f"{target_width}x{target_height}"])
        ffmpeg_cmd_read.extend([
            '-r', fps,
            '-vsync', '1',
            '-f', 'rawvideo',
            '-pix_fmt', 'yuv420p',
            '-vcodec', 'rawvideo',
            '-an', '-',
        ])

        ffmpeg_read_process = subprocess.Popen(
            ffmpeg_cmd_read,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=8 * 1024 * 1024,
        )

        # Пул предсозданных YUV420p кадров для zero-allocation чтения (в 2 раза меньше трафика и RAM, чем BGR24!)
        frame_size = (target_width * target_height * 3) // 2
        pool_size = DETECT_EVERY_N_FRAMES + 1
        frame_pool = [np.empty(frame_size, dtype=np.uint8) for _ in range(pool_size)]
        pool_idx = 0

        buffer = []
        faces_start = []
        is_first_chunk = True
        cancelled = False

        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break

                frame_array = frame_pool[pool_idx]
                frame_view = memoryview(frame_array)

                bytes_read = 0
                while bytes_read < frame_size:
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        break
                    chunk = ffmpeg_read_process.stdout.readinto(frame_view[bytes_read:])
                    if not chunk:
                        break
                    bytes_read += chunk

                if cancelled:
                    break

                if bytes_read != frame_size:
                    if buffer:
                        last_frame = buffer[-1]
                        faces_end = self._detect_faces(last_frame)
                        if is_first_chunk:
                            faces_start = self._detect_faces(buffer[0])

                        matched_pairs = self._match_faces(faces_start, faces_end, target_width, target_height)

                        for i, b_frame in enumerate(buffer):
                            if cancel_event is not None and cancel_event.is_set():
                                cancelled = True
                                break
                            alpha = i / max(1, len(buffer) - 1) if len(buffer) > 1 else 0
                            for (sf, ef) in matched_pairs:
                                x = int(sf[0] * (1 - alpha) + ef[0] * alpha)
                                y = int(sf[1] * (1 - alpha) + ef[1] * alpha)
                                w = int(sf[2] * (1 - alpha) + ef[2] * alpha)
                                h = int(sf[3] * (1 - alpha) + ef[3] * alpha)
                                self._apply_blur(b_frame, x, y, w, h, target_width, target_height)
                            try:
                                write_process.stdin.write(memoryview(b_frame))
                            except Exception:
                                pass
                    break

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
                        if cancel_event is not None and cancel_event.is_set():
                            cancelled = True
                            break
                        if is_first_chunk:
                            alpha = i / max(1, len(buffer) - 1)
                        else:
                            alpha = (i + 1) / len(buffer)

                        for (sf, ef) in matched_pairs:
                            x = int(sf[0] * (1 - alpha) + ef[0] * alpha)
                            y = int(sf[1] * (1 - alpha) + ef[1] * alpha)
                            w = int(sf[2] * (1 - alpha) + ef[2] * alpha)
                            h = int(sf[3] * (1 - alpha) + ef[3] * alpha)
                            self._apply_blur(b_frame, x, y, w, h, target_width, target_height)
                        try:
                            write_process.stdin.write(memoryview(b_frame))
                        except Exception:
                            pass

                    if cancelled:
                        break

                    is_first_chunk = False

                    faces_start = faces_end
                    buffer = []
        finally:
            try:
                ffmpeg_read_process.stdout.close()
            except Exception:
                pass
            if cancelled:
                ffmpeg_read_process.terminate()
            ffmpeg_read_process.wait()

            if 'write_process' in locals() and write_process:
                try:
                    if write_process.stdin:
                        write_process.stdin.close()
                except Exception:
                    pass
                if cancelled:
                    write_process.terminate()
                write_process.wait()

        if cancelled:
            return False
        return write_process.returncode == 0

