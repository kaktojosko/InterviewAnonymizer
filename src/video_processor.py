import cv2
import numpy as np
import os
import subprocess
from fractions import Fraction
from src.config import (
    ADAPTIVE_MIN_DETECT_STEP,
    DETECT_EVERY_N_FRAMES,
    FACE_PADDING_PERCENT_X,
    FACE_PADDING_PERCENT_Y,
    MOTION_CHANGED_RATIO_TRIGGER,
    MOTION_DIFF_PIXEL_THRESHOLD,
    OUTPUT_VIDEO_CRF,
    OUTPUT_VIDEO_PRESET,
    YUNET_MIN_CONFIDENCE,
    YUNET_TARGET_HEIGHT,
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

        # Даунскейл до YUNET_TARGET_HEIGHT (576p) ТОЛЬКО для нейросети:
        # сохраняет достаточный размер мелких лиц в галерее видеозвонков (~16-24 px+)
        # при высокой скорости инференса на CPU.
        target_height = int(YUNET_TARGET_HEIGHT)
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

            # Preallocate ML-resolution I420 + BGR buffers so _detect_faces does zero heap allocations
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
        Supports planar YUV420p (1-D uint8 buffer) with ML-resolution color conversion,
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

    @staticmethod
    def _extract_motion_thumb(frame, width: int, height: int) -> np.ndarray:
        """
        Extracts a tiny (~64x36) uint8 luma thumbnail via zero-copy strided slicing
        (plus a 2 KB copy so frame_pool reuse never mutates cached thumbnails).
        """
        if frame.ndim == 1:
            y_plane = frame[: width * height].reshape((height, width))
        else:
            height, width = frame.shape[:2]
            # Green channel is an ultra-fast proxy for luminance on BGR frames
            y_plane = frame[:, :, 1]
        sy = max(1, height // 36)
        sx = max(1, width // 64)
        return y_plane[::sy, ::sx].copy()

    @staticmethod
    def _thumb_changed_ratio(thumb_a: np.ndarray, thumb_b: np.ndarray) -> float:
        if thumb_a is None or thumb_b is None or thumb_a.shape != thumb_b.shape or thumb_a.size == 0:
            return 1.0
        diff = np.abs(thumb_a.astype(np.int16) - thumb_b.astype(np.int16))
        return float(np.mean(diff > MOTION_DIFF_PIXEL_THRESHOLD))

    @staticmethod
    def _local_region_changed_ratio(
        thumb_a: np.ndarray,
        thumb_b: np.ndarray,
        face: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> float:
        if thumb_a is None or thumb_b is None or thumb_a.shape != thumb_b.shape or thumb_a.size == 0:
            return 1.0
        th, tw = thumb_a.shape[:2]
        x, y, w, h = face
        tx1 = max(0, min(tw - 1, int(x * tw / max(1, width))))
        ty1 = max(0, min(th - 1, int(y * th / max(1, height))))
        tx2 = max(tx1 + 1, min(tw, int((x + w) * tw / max(1, width)) + 1))
        ty2 = max(ty1 + 1, min(th, int((y + h) * th / max(1, height)) + 1))
        patch_a = thumb_a[ty1:ty2, tx1:tx2]
        patch_b = thumb_b[ty1:ty2, tx1:tx2]
        if patch_a.size == 0:
            return 1.0
        diff = np.abs(patch_a.astype(np.int16) - patch_b.astype(np.int16))
        return float(np.mean(diff > MOTION_DIFF_PIXEL_THRESHOLD))

    def _match_faces_detailed(self, faces_start, faces_end, width, height):
        """
        Globally-optimal (cost-sorted) 1-to-1 face matching between two keyframes.
        Enforces size compatibility so large main speakers never steal small gallery tiles,
        while allowing wide window-drag displacements (up to 35% of screen width).

        Returns: (matched_pairs, unmatched_start, unmatched_end)
        """
        candidates = []
        for i, sf in enumerate(faces_start):
            cx1 = sf[0] + sf[2] * 0.5
            cy1 = sf[1] + sf[3] * 0.5
            sw = max(1.0, float(sf[2]))
            sh = max(1.0, float(sf[3]))
            for j, ef in enumerate(faces_end):
                ew = max(1.0, float(ef[2]))
                eh = max(1.0, float(ef[3]))
                w_ratio = max(sw / ew, ew / sw)
                h_ratio = max(sh / eh, eh / sh)
                if w_ratio > 1.85 or h_ratio > 1.85:
                    continue
                cx2 = ef[0] + ef[2] * 0.5
                cy2 = ef[1] + ef[3] * 0.5
                dist = float(np.hypot(cx1 - cx2, cy1 - cy2))
                max_dim = max(sw, sh, ew, eh)
                max_allowed = max(max_dim * 4.5, float(width) * 0.35)
                if dist <= max_allowed:
                    size_pen = 0.5 * (abs(ew - sw) + abs(eh - sh))
                    candidates.append((dist + size_pen, i, j))

        candidates.sort(key=lambda item: item[0])
        used_s = set()
        used_e = set()
        matched_pairs = []
        for _cost, i, j in candidates:
            if i in used_s or j in used_e:
                continue
            used_s.add(i)
            used_e.add(j)
            matched_pairs.append((faces_start[i], faces_end[j]))

        unmatched_start = [sf for i, sf in enumerate(faces_start) if i not in used_s]
        unmatched_end = [ef for j, ef in enumerate(faces_end) if j not in used_e]
        return matched_pairs, unmatched_start, unmatched_end

    def _match_faces(self, faces_start, faces_end, width, height):
        """
        Matches faces between two keyframes for interpolation.
        Returns a list of tuples: (start_face, end_face), keeping unmatched faces
        covered at their detected coordinates.
        """
        matched_pairs, unmatched_start, unmatched_end = self._match_faces_detailed(
            faces_start, faces_end, width, height
        )
        out = list(matched_pairs)
        for sf in unmatched_start:
            out.append((sf, sf))
        for ef in unmatched_end:
            out.append((ef, ef))
        return out

    def _bridge_dropout_keypoints(self, keypoints: dict, width: int, height: int) -> None:
        """
        Reconstructs isolated 1-keyframe YuNet drop-outs (e.g. momentary motion blur
        during a fast window drag or quick blink/tilt) when the same face is detected
        at both neighboring keyframes k_{j-1} and k_{j+1}.
        """
        sorted_keys = sorted(keypoints.keys())
        if len(sorted_keys) < 3:
            return

        for idx in range(1, len(sorted_keys) - 1):
            k_prev = sorted_keys[idx - 1]
            k_mid = sorted_keys[idx]
            k_next = sorted_keys[idx + 1]
            f_prev = keypoints[k_prev]
            f_mid = keypoints[k_mid]
            f_next = keypoints[k_next]
            if not f_prev or not f_next:
                continue

            span = max(1, k_next - k_prev)
            alpha = (k_mid - k_prev) / float(span)
            bridged_pairs, _, _ = self._match_faces_detailed(f_prev, f_next, width, height)
            for sf, ef in bridged_pairs:
                mx = int(sf[0] * (1.0 - alpha) + ef[0] * alpha)
                my = int(sf[1] * (1.0 - alpha) + ef[1] * alpha)
                mw = int(sf[2] * (1.0 - alpha) + ef[2] * alpha)
                mh = int(sf[3] * (1.0 - alpha) + ef[3] * alpha)
                mcx = mx + mw * 0.5
                mcy = my + mh * 0.5
                # Only inject if no face already covers the expected midpoint
                covered = False
                for existing in f_mid:
                    ecx = existing[0] + existing[2] * 0.5
                    ecy = existing[1] + existing[3] * 0.5
                    tol = max(20.0, 0.85 * max(mw, mh, existing[2], existing[3]))
                    if np.hypot(mcx - ecx, mcy - ecy) <= tol:
                        covered = True
                        break
                if not covered:
                    f_mid.append((mx, my, mw, mh))

    def _resolve_batch_keypoints(
        self,
        buffer: list,
        prev_faces,
        prev_last_thumb,
        width: int,
        height: int,
        cancel_event=None,
    ):
        """
        Resolves face detections for a batch of frames (up to DETECT_EVERY_N_FRAMES=18).
        On calm interview intervals, runs only 1 detection per 18 frames.
        When window dragging, resizing, layout/tab switching, face appearance, or rapid
        face motion occurs, recursively bisects the interval down to ADAPTIVE_MIN_DETECT_STEP
        (2 frames) so faces never leak across motion gaps.

        Returns: (keypoints_dict, end_faces_raw, last_thumb, sticky_overlays)
        """
        n = len(buffer)
        if n == 0:
            return {}, [], prev_last_thumb, []

        thumbs: dict[int, np.ndarray] = {}

        def get_thumb(idx: int) -> np.ndarray:
            t = thumbs.get(idx)
            if t is None:
                t = self._extract_motion_thumb(buffer[idx], width, height)
                thumbs[idx] = t
            return t

        keypoints: dict[int, list] = {}
        zero_is_fresh = False
        boundary_cut = False

        if prev_faces is None or prev_last_thumb is None:
            keypoints[0] = self._detect_faces(buffer[0])
            zero_is_fresh = True
        else:
            boundary_diff = self._thumb_changed_ratio(prev_last_thumb, get_thumb(0))
            if boundary_diff >= MOTION_CHANGED_RATIO_TRIGGER:
                keypoints[0] = self._detect_faces(buffer[0])
                zero_is_fresh = True
                boundary_cut = True
            else:
                keypoints[0] = list(prev_faces)

        if n == 1:
            end_raw = list(keypoints[0])
            return keypoints, end_raw, get_thumb(0), []

        last_idx = n - 1
        keypoints[last_idx] = self._detect_faces(buffer[last_idx])
        end_raw = list(keypoints[last_idx])

        min_step = max(1, int(ADAPTIVE_MIN_DETECT_STEP))

        def _needs_subdivision(lo: int, hi: int) -> bool:
            if hi - lo <= min_step:
                return False

            f_lo = keypoints[lo]
            f_hi = keypoints[hi]

            # 1. Face count changed (someone appeared, left, or flickered)
            if len(f_lo) != len(f_hi):
                return True

            # 2. Global screen/window motion (browser drag, resize, scroll, tab switch)
            t_lo = get_thumb(lo)
            t_hi = get_thumb(hi)
            if self._thumb_changed_ratio(t_lo, t_hi) >= MOTION_CHANGED_RATIO_TRIGGER:
                return True

            # Also probe midpoint thumbnail on spans >= 6 to catch mid-interval window flashes
            if hi - lo >= 6:
                mid_probe = (lo + hi) // 2
                t_mid = get_thumb(mid_probe)
                if (
                    self._thumb_changed_ratio(t_lo, t_mid) >= MOTION_CHANGED_RATIO_TRIGGER
                    or self._thumb_changed_ratio(t_mid, t_hi) >= MOTION_CHANGED_RATIO_TRIGGER
                ):
                    return True

            # 3. Face displacement or scale jump
            if f_lo or f_hi:
                matched_pairs, unm_s, unm_e = self._match_faces_detailed(f_lo, f_hi, width, height)
                if unm_s or unm_e:
                    return True
                for sf, ef in matched_pairs:
                    dx = abs((ef[0] + ef[2] * 0.5) - (sf[0] + sf[2] * 0.5))
                    dy = abs((ef[1] + ef[3] * 0.5) - (sf[1] + sf[3] * 0.5))
                    min_dim = max(16.0, float(min(sf[2], sf[3], ef[2], ef[3])))
                    if max(dx, dy) > max(24.0, 0.35 * min_dim):
                        return True
                    w_ratio = max(sf[2] / max(1, ef[2]), ef[2] / max(1, sf[2]))
                    h_ratio = max(sf[3] / max(1, ef[3]), ef[3] / max(1, sf[3]))
                    if max(w_ratio, h_ratio) > 1.25:
                        return True

            return False

        def _subdivide(lo: int, hi: int) -> None:
            nonlocal zero_is_fresh
            if cancel_event is not None and cancel_event.is_set():
                return
            if not _needs_subdivision(lo, hi):
                return
            if lo == 0 and not zero_is_fresh:
                keypoints[0] = self._detect_faces(buffer[0])
                zero_is_fresh = True
            mid = (lo + hi) // 2
            if mid not in keypoints:
                keypoints[mid] = self._detect_faces(buffer[mid])
            _subdivide(lo, mid)
            _subdivide(mid, hi)

        _subdivide(0, last_idx)

        # Bridge isolated 1-keyframe drop-outs (e.g. motion blur during drag)
        self._bridge_dropout_keypoints(keypoints, width, height)

        # 1-cycle static-region anti-flicker persistence: if a previously detected face
        # temporarily dropped (e.g. participant briefly turned into profile / hand over mouth)
        # while the window itself stayed stationary, keep that static tile blurred for this cycle.
        sticky_overlays = []
        if prev_faces and not boundary_cut:
            global_diff = self._thumb_changed_ratio(prev_last_thumb, get_thumb(last_idx))
            if global_diff < MOTION_CHANGED_RATIO_TRIGGER:
                _, dropped_prev, _ = self._match_faces_detailed(prev_faces, end_raw, width, height)
                for df in dropped_prev:
                    local_diff = self._local_region_changed_ratio(
                        prev_last_thumb, get_thumb(last_idx), df, width, height
                    )
                    if local_diff < 0.30:
                        sticky_overlays.append(df)

        return keypoints, end_raw, get_thumb(last_idx), sticky_overlays

    def _apply_blur(
        self,
        frame,
        x,
        y,
        w,
        h,
        frame_width: int = 0,
        frame_height: int = 0,
        extra_pad_x: int = 0,
        extra_pad_y: int = 0,
    ):
        """Applies a strong pixelation (mosaic) effect in-place on planar YUV420p or BGR frames."""
        pad_x = int(w * FACE_PADDING_PERCENT_X) + max(0, int(extra_pad_x))
        pad_y = int(h * FACE_PADDING_PERCENT_Y) + max(0, int(extra_pad_y))

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

    def _blur_and_write_buffer(
        self,
        buffer: list,
        keypoints: dict,
        sticky_overlays: list,
        target_width: int,
        target_height: int,
        write_process,
        cancel_event=None,
    ) -> bool:
        """
        Interpolates faces piecewise across all adaptive keypoints in `buffer`,
        applies motion-swept mosaic padding in-place, and writes every frame to FFmpeg.
        Returns True if cancelled mid-buffer, False otherwise.
        """
        sorted_keys = sorted(keypoints.keys())
        if not sorted_keys:
            for b_frame in buffer:
                if cancel_event is not None and cancel_event.is_set():
                    return True
                for ov in sticky_overlays:
                    self._apply_blur(b_frame, ov[0], ov[1], ov[2], ov[3], target_width, target_height)
                try:
                    write_process.stdin.write(memoryview(b_frame))
                except Exception:
                    pass
            return False

        if len(sorted_keys) == 1:
            faces_only = keypoints[sorted_keys[0]]
            for b_frame in buffer:
                if cancel_event is not None and cancel_event.is_set():
                    return True
                for f in faces_only:
                    self._apply_blur(b_frame, f[0], f[1], f[2], f[3], target_width, target_height)
                for ov in sticky_overlays:
                    self._apply_blur(b_frame, ov[0], ov[1], ov[2], ov[3], target_width, target_height)
                try:
                    write_process.stdin.write(memoryview(b_frame))
                except Exception:
                    pass
            return False

        num_segs = len(sorted_keys) - 1
        for seg_idx in range(num_segs):
            k_a = sorted_keys[seg_idx]
            k_b = sorted_keys[seg_idx + 1]
            span = max(1, k_b - k_a)
            matched_pairs = self._match_faces(keypoints[k_a], keypoints[k_b], target_width, target_height)

            end_frame = k_b + 1 if seg_idx == num_segs - 1 else k_b
            for i in range(k_a, end_frame):
                if cancel_event is not None and cancel_event.is_set():
                    return True
                b_frame = buffer[i]
                alpha = (i - k_a) / float(span)
                mid_weight = 4.0 * alpha * (1.0 - alpha)

                for sf, ef in matched_pairs:
                    x = int(sf[0] * (1.0 - alpha) + ef[0] * alpha)
                    y = int(sf[1] * (1.0 - alpha) + ef[1] * alpha)
                    w = int(sf[2] * (1.0 - alpha) + ef[2] * alpha)
                    h = int(sf[3] * (1.0 - alpha) + ef[3] * alpha)
                    step_dx = abs((ef[0] + ef[2] * 0.5) - (sf[0] + sf[2] * 0.5))
                    step_dy = abs((ef[1] + ef[3] * 0.5) - (sf[1] + sf[3] * 0.5))
                    extra_px = int(min(w * 0.65, step_dx * 0.35 * mid_weight))
                    extra_py = int(min(h * 0.65, step_dy * 0.35 * mid_weight))
                    self._apply_blur(
                        b_frame,
                        x,
                        y,
                        w,
                        h,
                        target_width,
                        target_height,
                        extra_pad_x=extra_px,
                        extra_pad_y=extra_py,
                    )

                for ov in sticky_overlays:
                    self._apply_blur(b_frame, ov[0], ov[1], ov[2], ov[3], target_width, target_height)

                try:
                    write_process.stdin.write(memoryview(b_frame))
                except Exception:
                    pass

        return False

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
        prev_faces = None
        prev_last_thumb = None
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
                        keypoints, _end_faces, _last_thumb, sticky = self._resolve_batch_keypoints(
                            buffer,
                            prev_faces,
                            prev_last_thumb,
                            target_width,
                            target_height,
                            cancel_event=cancel_event,
                        )
                        cancelled = self._blur_and_write_buffer(
                            buffer,
                            keypoints,
                            sticky,
                            target_width,
                            target_height,
                            write_process,
                            cancel_event=cancel_event,
                        )
                    break

                frame = frame_array
                buffer.append(frame)
                pool_idx = (pool_idx + 1) % pool_size

                if len(buffer) >= DETECT_EVERY_N_FRAMES:
                    keypoints, end_faces, last_thumb, sticky = self._resolve_batch_keypoints(
                        buffer,
                        prev_faces,
                        prev_last_thumb,
                        target_width,
                        target_height,
                        cancel_event=cancel_event,
                    )
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        break

                    cancelled = self._blur_and_write_buffer(
                        buffer,
                        keypoints,
                        sticky,
                        target_width,
                        target_height,
                        write_process,
                        cancel_event=cancel_event,
                    )
                    if cancelled:
                        break

                    prev_faces = end_faces
                    prev_last_thumb = last_thumb
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


