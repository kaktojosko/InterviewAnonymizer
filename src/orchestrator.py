import os
import shutil
import time
import glob
import multiprocessing as mp
import threading
import wave
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait as futures_wait,
)
from src.cancellation import CancellationToken, TaskCancelledError
from src.io_manager import IOManager
from src.ffmpeg_utils import FFmpegUtils
from src.audio_processor import AudioProcessor
from src.video_processor import VideoProcessor
from src.config import (
    ASR_MODEL_PROFILES,
    CHUNK_DURATION_SECONDS,
    CPU_COUNT,
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_ASR_MODEL,
    DEFAULT_COMPUTE_DEVICE,
    MAX_ALLOWED_WORKERS,
    MAX_WORKERS,
    TRANSCRIPTION_ENABLED_DEFAULT,
)

# Глобальные объекты воркера (инициализируются один раз в каждом spawn-процессе)
_worker_vp = None
_worker_log_queue = None
_worker_cancel_event = None


def init_worker(log_queue=None, cancel_event=None):
    global _worker_vp, _worker_log_queue, _worker_cancel_event
    _worker_log_queue = log_queue
    _worker_cancel_event = cancel_event
    _worker_vp = VideoProcessor()


def process_chunk(
    input_chunk: str,
    output_chunk: str,
    fps_str: str,
    width: int,
    height: int,
    enc_threads: int = 2,
    log_queue=None,
    cancel_event=None,
) -> tuple[bool, float]:
    global _worker_vp, _worker_log_queue, _worker_cancel_event
    ev = cancel_event if cancel_event is not None else _worker_cancel_event
    if ev is not None and ev.is_set():
        return False, 0.0
    q = log_queue if log_queue is not None else _worker_log_queue
    if q is not None:
        q.put(f"[Worker] Started: {os.path.basename(input_chunk)}")
    start_time = time.time()
    if _worker_vp is None:
        _worker_vp = VideoProcessor()
    success = _worker_vp.process(
        input_chunk,
        output_chunk,
        fps=fps_str,
        orig_width=width,
        orig_height=height,
        enc_threads=enc_threads,
        cancel_event=ev,
    )
    return success, time.time() - start_time


class Orchestrator:
    @staticmethod
    def _save_cached_audio(src_wav_path: str, dst_cache_path: str | None) -> bool:
        if not dst_cache_path or not os.path.isfile(src_wav_path):
            return False
        if os.path.abspath(src_wav_path) == os.path.abspath(dst_cache_path):
            return True
        try:
            os.makedirs(os.path.dirname(os.path.abspath(dst_cache_path)), exist_ok=True)
            tmp_dst = f"{dst_cache_path}.part"
            shutil.copy2(src_wav_path, tmp_dst)
            os.replace(tmp_dst, dst_cache_path)
            return os.path.isfile(dst_cache_path)
        except OSError:
            return False

    @staticmethod
    def _resolve_post_video_threads(workers: int) -> int:
        # Once video blurring is complete (or in transcription-only mode), all CPU
        # cores are free: respect explicit low-CPU caps (1..3), otherwise use
        # available cores up to MAX_ALLOWED_WORKERS.
        return (
            workers
            if workers < 4
            else max(workers, min(MAX_ALLOWED_WORKERS, max(4, CPU_COUNT - 2)))
        )

    @staticmethod
    def process_video_job(
        input_path: str,
        output_path: str,
        job_dir: str,
        status_cb=None,
        worker_count: int | None = None,
        transcribe: bool = TRANSCRIPTION_ENABLED_DEFAULT,
        transcript_path: str | None = None,
        asr_model: str = DEFAULT_ASR_MODEL,
        speaker_count: str = "auto",
        compute_device: str = DEFAULT_COMPUTE_DEVICE,
        cancel_token: CancellationToken | None = None,
        language: str = DEFAULT_ASR_LANGUAGE,
        cached_audio_path: str | None = None,
    ) -> dict:
        log_lock = threading.Lock()

        def update(msg):
            if cancel_token is not None and cancel_token.is_cancelled():
                return
            with log_lock:
                print(msg)
                if status_cb:
                    status_cb(msg)

        if cancel_token is not None:
            cancel_token.check()

        update(f"Starting processing for: {input_path}")
        total_start_time = time.time()
        workers = max(1, worker_count or MAX_WORKERS)
        requested_device = (compute_device or DEFAULT_COMPUTE_DEVICE).strip().lower()

        # Pre-encode pitch-shifted audio directly to AAC (.m4a) in the background
        # during video blurring so final muxing is an instant stream copy!
        audio_pitched_path = os.path.join(job_dir, "audio_pitched.m4a")
        audio_16k_path = os.path.join(job_dir, "audio_16k.wav")
        if transcribe and not transcript_path:
            stem, _ = os.path.splitext(output_path)
            transcript_path = f"{stem}_transcript.txt"

        # 1. Probe (Metadata)
        update("Анализ видеофайла...")
        t0 = time.time()
        width, height, fps_str, duration, has_audio, sample_rate = FFmpegUtils.get_video_metadata(input_path)
        if cancel_token is not None:
            cancel_token.check()
        update(f"[Profile] Probe finished in {time.time() - t0:.2f}s")
        update(f"Целевой FPS обработки: {fps_str}")

        profile_info = ASR_MODEL_PROFILES.get(asr_model, ASR_MODEL_PROFILES[DEFAULT_ASR_MODEL])
        use_parallel_gpu = False
        if (
            has_audio
            and transcribe
            and transcript_path
            and requested_device == "cuda"
            and profile_info.get("engine") == "faster_whisper"
        ):
            from src.transcriber import probe_cuda_available

            use_parallel_gpu = probe_cuda_available()

        # 2. Split Video & Parallel Audio / GPU Whisper ASR
        audio_future = None
        transcribe_extract_future = None
        parallel_gpu_future = None
        need_raw_wav = bool(has_audio and ((transcribe and transcript_path) or cached_audio_path))
        audio_executor = ThreadPoolExecutor(max_workers=2)
        try:
            # Split video first so 3 concurrent FFmpeg readers don't thrash the disk
            update("Разделение видео для параллельной обработки...")
            t0 = time.time()

            if duration > 0:
                adaptive_chunk_time = max(30, min(90, int(duration / workers)))
            else:
                adaptive_chunk_time = max(30, CHUNK_DURATION_SECONDS)

            update(f"Оптимальный размер чанка: {adaptive_chunk_time}с (Длительность: {duration:.1f}с)")

            chunks_dir = os.path.join(job_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            chunk_pattern = os.path.join(chunks_dir, "chunk_%04d.mp4")
            FFmpegUtils.split_video(
                input_path, chunk_pattern, adaptive_chunk_time, cancel_token=cancel_token
            )
            if cancel_token is not None:
                cancel_token.check()
            update(f"[Profile] Split video finished in {time.time() - t0:.2f}s")

            if has_audio:
                update("Запуск обработки аудио (в фоне)...")
                audio_future = audio_executor.submit(
                    AudioProcessor.process, input_path, audio_pitched_path, sample_rate, cancel_token
                )
                if need_raw_wav:
                    if use_parallel_gpu:
                        def _extract_and_run_gpu_asr() -> tuple[bool, dict | None]:
                            raw_ok = FFmpegUtils.extract_transcription_audio(
                                input_path, audio_16k_path, cancel_token=cancel_token
                            )
                            if not raw_ok:
                                return False, None
                            if cancel_token is not None:
                                cancel_token.check()
                            update("Параллельная транскрибация на GPU (в фоне обработки видео)...")
                            from src.transcriber import Transcriber

                            asr_state = Transcriber.transcribe_units_only(
                                wav_path=audio_16k_path,
                                asr_model=asr_model,
                                compute_device="cuda",
                                threads=1,
                                status_cb=update,
                                cancel_token=cancel_token,
                                language=language,
                            )
                            return True, asr_state

                        parallel_gpu_future = audio_executor.submit(_extract_and_run_gpu_asr)
                    else:
                        transcribe_extract_future = audio_executor.submit(
                            FFmpegUtils.extract_transcription_audio,
                            input_path,
                            audio_16k_path,
                            cancel_token,
                        )

            input_chunks = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.mp4")))
            list_file_path = None
            video_blurred_path = None

            if not input_chunks:
                # Fallback for clips where segmenter produced 0 chunks
                update("Распознавание лиц и блюр (Быстрый режим для короткого видео)...")
                video_blurred_path = os.path.join(job_dir, "video_blurred.mp4")
                vp = VideoProcessor()
                fallback_ev = threading.Event()
                if cancel_token is not None:
                    cancel_token.register_cancel_callback(fallback_ev.set)
                enc_threads = max(1, min(4, CPU_COUNT))
                ok = vp.process(
                    input_path,
                    video_blurred_path,
                    fps=fps_str,
                    orig_width=width,
                    orig_height=height,
                    enc_threads=enc_threads,
                    cancel_event=fallback_ev,
                )
                if cancel_token is not None:
                    cancel_token.check()
                if not ok:
                    raise RuntimeError("Ошибка обработки видео")
            else:
                # 3. Process chunks in parallel via 'spawn' (avoids Linux fork() deadlocks with CUDA / ONNX Runtime)
                active_workers = max(1, min(workers, len(input_chunks)))
                enc_threads = max(1, min(4, CPU_COUNT // active_workers))
                update(f"Распознавание лиц и блюр (Потоков: {active_workers})...")
                t0 = time.time()
                processed_chunks_dir = os.path.join(job_dir, "processed_chunks")
                os.makedirs(processed_chunks_dir, exist_ok=True)

                spawn_ctx = mp.get_context("spawn")
                log_queue = spawn_ctx.Queue()
                worker_cancel_event = spawn_ctx.Event()

                def log_listener():
                    while True:
                        try:
                            msg = log_queue.get()
                            if msg == "DONE":
                                break
                            update(msg)
                        except Exception:
                            break

                listener_thread = threading.Thread(target=log_listener, daemon=True)
                listener_thread.start()

                futures_map = {}
                executor = ProcessPoolExecutor(
                    max_workers=active_workers,
                    mp_context=spawn_ctx,
                    initializer=init_worker,
                    initargs=(log_queue, worker_cancel_event),
                )

                def _abort_workers():
                    worker_cancel_event.set()
                    for f in list(futures_map.keys()):
                        f.cancel()

                if cancel_token is not None:
                    cancel_token.register_cancel_callback(_abort_workers)

                try:
                    for idx, chunk in enumerate(input_chunks):
                        if cancel_token is not None:
                            cancel_token.check()
                        out_chunk = os.path.join(processed_chunks_dir, f"out_{idx:04d}.mp4")
                        f = executor.submit(
                            process_chunk, chunk, out_chunk, fps_str, width, height, enc_threads
                        )
                        futures_map[f] = out_chunk

                    output_chunks = []
                    completed_count = 0
                    total_chunks = len(futures_map)
                    for future in as_completed(futures_map):
                        if cancel_token is not None:
                            cancel_token.check()
                        out_chunk = futures_map[future]
                        chunk_ok, chunk_duration = future.result()
                        if cancel_token is not None:
                            cancel_token.check()
                        if not chunk_ok:
                            raise RuntimeError(f"Ошибка обработки видео-чанка: {os.path.basename(out_chunk)}")
                        completed_count += 1
                        update(
                            f"[Profile] Finished: {os.path.basename(out_chunk)} "
                            f"({completed_count}/{total_chunks}) in {chunk_duration:.2f}s"
                        )
                        output_chunks.append(out_chunk)
                except BaseException:
                    worker_cancel_event.set()
                    executor.shutdown(wait=False, cancel_futures=True)
                    procs = list((getattr(executor, "_processes", None) or {}).values())
                    deadline = time.time() + 0.6
                    for p in procs:
                        rem = max(0.0, deadline - time.time())
                        if p.is_alive() and rem > 0:
                            p.join(timeout=rem)
                        if p.is_alive():
                            try:
                                p.terminate()
                            except Exception:
                                pass
                    raise
                else:
                    executor.shutdown(wait=True)
                finally:
                    try:
                        log_queue.put("DONE")
                    except Exception:
                        pass
                    listener_thread.join(timeout=1.5)
                    try:
                        log_queue.close()
                    except Exception:
                        pass

                update(f"[Profile] All chunks processed in {time.time() - t0:.2f}s")

                list_file_path = os.path.join(job_dir, "concat_list.txt")
                with open(list_file_path, "w") as f:
                    for out_chunk in sorted(output_chunks):
                        safe_path = out_chunk.replace('\\', '/')
                        f.write(f"file '{safe_path}'\n")

            # 4. Wait for background pitch-shifted AAC audio
            if cancel_token is not None:
                cancel_token.check()
            final_audio_path = None
            raw_wav_ready = False
            if has_audio and audio_future:
                update("Ожидание завершения обработки аудио...")
                t0 = time.time()
                audio_success = audio_future.result()
                update(f"[Profile] Audio wait finished in {time.time() - t0:.2f}s")
                if audio_success:
                    final_audio_path = audio_pitched_path
            if transcribe_extract_future is not None:
                raw_wav_ready = bool(transcribe_extract_future.result())

            # 5. Single-pass stream-copy concat & mux of final MP4
            if cancel_token is not None:
                cancel_token.check()
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            t0 = time.time()
            if list_file_path is not None:
                update("Склейка чанков и сборка финального видео...")
                FFmpegUtils.concat_and_mux(
                    list_file_path, final_audio_path, output_path, cancel_token=cancel_token
                )
            else:
                update("Сборка финального видео...")
                FFmpegUtils.mux(
                    video_blurred_path,
                    final_audio_path,
                    output_path,
                    cancel_token=cancel_token,
                    video_is_h264=True,
                )
            update(f"[Profile] Mux finished in {time.time() - t0:.2f}s")

            # 6. Local Auto-Transcription with Speaker Diarization (if enabled)
            transcript_ready = False
            audio_cached = False
            if transcribe and transcript_path:
                if cancel_token is not None:
                    cancel_token.check()
                from src.transcriber import Transcriber, format_transcript_txt

                post_video_threads = Orchestrator._resolve_post_video_threads(workers)

                gpu_asr_state = None
                if parallel_gpu_future is not None:
                    if not parallel_gpu_future.done():
                        update("Ожидание завершения распознавания речи на GPU...")
                    raw_wav_ready, gpu_asr_state = parallel_gpu_future.result()

                if raw_wav_ready and cached_audio_path:
                    audio_cached = Orchestrator._save_cached_audio(audio_16k_path, cached_audio_path)

                if not has_audio or not raw_wav_ready:
                    update("Аудиодорожка отсутствует — формирование пустого файла транскрибации...")
                    os.makedirs(os.path.dirname(os.path.abspath(transcript_path)), exist_ok=True)
                    empty_txt = format_transcript_txt(
                        turns=[],
                        duration_sec=duration,
                        model_name=profile_info["short_name"],
                        num_speakers=0,
                        note="[Аудиодорожка в исходном видео отсутствует]",
                    )
                    with open(transcript_path, "w", encoding="utf-8") as f:
                        f.write(empty_txt)
                    transcript_ready = True
                elif parallel_gpu_future is not None and gpu_asr_state is not None:
                    diar_cpu_threads = min(8, max(post_video_threads, MAX_ALLOWED_WORKERS))
                    Transcriber.diarize_and_write_transcript(
                        asr_state=gpu_asr_state,
                        output_txt_path=transcript_path,
                        speaker_count=speaker_count,
                        threads=diar_cpu_threads,
                        status_cb=update,
                        cancel_token=cancel_token,
                    )
                    transcript_ready = os.path.isfile(transcript_path)
                else:
                    update("Запуск локальной автотранскрибации по ролям...")
                    Transcriber.transcribe_audio_file(
                        wav_path=audio_16k_path,
                        output_txt_path=transcript_path,
                        asr_model=asr_model,
                        speaker_count=speaker_count,
                        threads=post_video_threads,
                        compute_device=requested_device,
                        status_cb=update,
                        cancel_token=cancel_token,
                        language=language,
                    )
                    transcript_ready = os.path.isfile(transcript_path)
            elif raw_wav_ready and cached_audio_path:
                audio_cached = Orchestrator._save_cached_audio(audio_16k_path, cached_audio_path)

        finally:
            if cancel_token is not None and cancel_token.is_cancelled():
                pending = [
                    f
                    for f in (audio_future, transcribe_extract_future, parallel_gpu_future)
                    if f is not None
                ]
                if pending:
                    futures_wait(pending, timeout=1.5)
                audio_executor.shutdown(wait=False, cancel_futures=True)
            else:
                audio_executor.shutdown(wait=True)

        if cancel_token is not None:
            cancel_token.check()

        update(f"Success! Output saved to: {output_path}. Total time: {time.time() - total_start_time:.2f}s")
        return {
            "video_ready": os.path.isfile(output_path),
            "transcript_ready": transcript_ready,
            "transcript_path": transcript_path if transcript_ready else None,
            "audio_cached": audio_cached,
        }

    @staticmethod
    def process_transcription_only_job(
        input_path: str,
        transcript_path: str,
        job_dir: str,
        status_cb=None,
        worker_count: int | None = None,
        asr_model: str = DEFAULT_ASR_MODEL,
        language: str = DEFAULT_ASR_LANGUAGE,
        speaker_count: str = "auto",
        compute_device: str = DEFAULT_COMPUTE_DEVICE,
        cancel_token: CancellationToken | None = None,
        cached_audio_path: str | None = None,
        input_is_wav_16k: bool = False,
        source_note: str | None = None,
    ) -> dict:
        """
        Runs speaker-diarized transcription directly on a video/audio file (or
        cached 16 kHz mono WAV) without splitting, face blurring, pitch-shifting,
        or video re-encoding. Used by 'Только транскрибация' mode and post-recording
        separate transcript generation.
        """
        log_lock = threading.Lock()

        def update(msg):
            if cancel_token is not None and cancel_token.is_cancelled():
                return
            with log_lock:
                print(msg)
                if status_cb:
                    status_cb(msg)

        if cancel_token is not None:
            cancel_token.check()

        total_start_time = time.time()
        workers = max(1, worker_count or MAX_WORKERS)
        requested_device = (compute_device or DEFAULT_COMPUTE_DEVICE).strip().lower()
        profile_info = ASR_MODEL_PROFILES.get(asr_model, ASR_MODEL_PROFILES[DEFAULT_ASR_MODEL])
        asr_threads = Orchestrator._resolve_post_video_threads(workers)

        if source_note:
            update(source_note)

        from src.transcriber import Transcriber, format_transcript_txt

        duration = 0.0
        has_audio = False
        wav_path: str | None = None
        audio_cached = False

        if input_is_wav_16k and os.path.isfile(input_path):
            update("Использование сохранённой оригинальной аудиодорожки (без искажения голоса)...")
            wav_path = input_path
            has_audio = True
            try:
                with wave.open(input_path, "rb") as wf:
                    fr = wf.getframerate() or 16000
                    duration = wf.getnframes() / float(fr)
            except Exception:
                duration = 0.0
            if cached_audio_path:
                audio_cached = Orchestrator._save_cached_audio(wav_path, cached_audio_path)
        else:
            update("Анализ медиафайла...")
            t0 = time.time()
            _, _, _, duration, has_audio, _ = FFmpegUtils.get_video_metadata(input_path)
            if cancel_token is not None:
                cancel_token.check()
            update(f"[Profile] Probe finished in {time.time() - t0:.2f}s (Длительность: {duration:.1f}с)")

            if has_audio:
                update("Извлечение аудиодорожки (16 кГц моно)...")
                t0 = time.time()
                extracted_wav = os.path.join(job_dir, "audio_16k.wav")
                raw_ok = FFmpegUtils.extract_transcription_audio(
                    input_path, extracted_wav, cancel_token=cancel_token
                )
                if cancel_token is not None:
                    cancel_token.check()
                update(f"[Profile] Audio extract finished in {time.time() - t0:.2f}s")
                if raw_ok:
                    wav_path = extracted_wav
                    if duration <= 0.0:
                        try:
                            with wave.open(extracted_wav, "rb") as wf:
                                fr = wf.getframerate() or 16000
                                duration = wf.getnframes() / float(fr)
                        except Exception:
                            pass
                    if cached_audio_path:
                        audio_cached = Orchestrator._save_cached_audio(extracted_wav, cached_audio_path)
                else:
                    has_audio = False

        if cancel_token is not None:
            cancel_token.check()

        os.makedirs(os.path.dirname(os.path.abspath(transcript_path)), exist_ok=True)
        # Write to a staging file inside job_dir first so cancelling a re-transcription
        # never corrupts or wipes an existing transcript file.
        staged_txt_path = os.path.join(job_dir, "transcript_staged.txt")

        if not has_audio or not wav_path:
            update("Аудиодорожка отсутствует — формирование пустого файла транскрибации...")
            empty_txt = format_transcript_txt(
                turns=[],
                duration_sec=duration,
                model_name=profile_info["short_name"],
                num_speakers=0,
                note="[Аудиодорожка в исходном файле отсутствует]",
            )
            with open(staged_txt_path, "w", encoding="utf-8") as f:
                f.write(empty_txt)
        else:
            update("Запуск локальной транскрибации по ролям...")
            Transcriber.transcribe_audio_file(
                wav_path=wav_path,
                output_txt_path=staged_txt_path,
                asr_model=asr_model,
                speaker_count=speaker_count,
                threads=asr_threads,
                compute_device=requested_device,
                status_cb=update,
                cancel_token=cancel_token,
                language=language,
            )

        if cancel_token is not None:
            cancel_token.check()

        shutil.copy2(staged_txt_path, f"{transcript_path}.part")
        os.replace(f"{transcript_path}.part", transcript_path)
        transcript_ready = os.path.isfile(transcript_path)

        update(
            f"Success! Transcript saved to: {transcript_path}. "
            f"Total time: {time.time() - total_start_time:.2f}s"
        )
        return {
            "video_ready": False,
            "transcript_ready": transcript_ready,
            "transcript_path": transcript_path if transcript_ready else None,
            "audio_cached": audio_cached,
        }

