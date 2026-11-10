import os
import time
import glob
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager
from src.io_manager import IOManager
from src.ffmpeg_utils import FFmpegUtils
from src.audio_processor import AudioProcessor
from src.video_processor import VideoProcessor
from src.config import (
    ASR_MODEL_PROFILES,
    CHUNK_DURATION_SECONDS,
    DEFAULT_ASR_MODEL,
    DEFAULT_COMPUTE_DEVICE,
    MAX_WORKERS,
    TRANSCRIPTION_ENABLED_DEFAULT,
)

# Глобальная переменная для воркера, чтобы не пересоздавать модель
_worker_vp = None

def init_worker():
    global _worker_vp
    _worker_vp = VideoProcessor()

def process_chunk(input_chunk: str, output_chunk: str, fps_str: str, width: int, height: int, log_queue=None) -> tuple[bool, float]:
    global _worker_vp
    if log_queue:
        log_queue.put(f"[Worker] Started: {os.path.basename(input_chunk)}")
    start_time = time.time()
    if _worker_vp is None:
        _worker_vp = VideoProcessor()
    success = _worker_vp.process(input_chunk, output_chunk, fps=fps_str, orig_width=width, orig_height=height)
    return success, time.time() - start_time

class Orchestrator:
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
    ) -> dict:
        log_lock = threading.Lock()

        def update(msg):
            with log_lock:
                print(msg)
                if status_cb:
                    status_cb(msg)
                
        update(f"Starting processing for: {input_path}")
        total_start_time = time.time()
        workers = max(1, worker_count or MAX_WORKERS)
        requested_device = (compute_device or DEFAULT_COMPUTE_DEVICE).strip().lower()
        
        audio_pitched_path = os.path.join(job_dir, "audio_pitched.wav")
        audio_16k_path = os.path.join(job_dir, "audio_16k.wav")
        if transcribe and not transcript_path:
            stem, _ = os.path.splitext(output_path)
            transcript_path = f"{stem}_transcript.txt"
        
        # 1. Probe (Metadata)
        update("Анализ видеофайла...")
        t0 = time.time()
        width, height, fps_str, duration, has_audio, sample_rate = FFmpegUtils.get_video_metadata(input_path)
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
        
        # 2. Parallel Audio & Split Video
        audio_future = None
        transcribe_extract_future = None
        parallel_gpu_future = None
        audio_executor = ThreadPoolExecutor(max_workers=2)
        try:
            if has_audio:
                update("Запуск обработки аудио (в фоне)...")
                audio_future = audio_executor.submit(AudioProcessor.process, input_path, audio_pitched_path, sample_rate)
                if transcribe and transcript_path:
                    if use_parallel_gpu:
                        def _extract_and_transcribe_on_gpu() -> tuple[bool, bool]:
                            raw_ok = FFmpegUtils.extract_transcription_audio(input_path, audio_16k_path)
                            if not raw_ok:
                                return False, False
                            update("Параллельная транскрибация на GPU (в фоне обработки видео)...")
                            from src.transcriber import Transcriber

                            Transcriber.transcribe_audio_file(
                                wav_path=audio_16k_path,
                                output_txt_path=transcript_path,
                                asr_model=asr_model,
                                speaker_count=speaker_count,
                                threads=workers,
                                diar_threads=max(1, min(2, workers)),
                                compute_device="cuda",
                                status_cb=update,
                            )
                            return True, os.path.isfile(transcript_path)

                        parallel_gpu_future = audio_executor.submit(_extract_and_transcribe_on_gpu)
                    else:
                        transcribe_extract_future = audio_executor.submit(
                            FFmpegUtils.extract_transcription_audio, input_path, audio_16k_path
                        )
                
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
            FFmpegUtils.split_video(input_path, chunk_pattern, adaptive_chunk_time)
            update(f"[Profile] Split video finished in {time.time() - t0:.2f}s")
            
            input_chunks = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.mp4")))
            
            if not input_chunks:
                # Fallback
                update("Распознавание лиц и блюр (Быстрый режим для короткого видео)...")
                video_blurred_path = os.path.join(job_dir, "video_blurred.mp4")
                vp = VideoProcessor()
                vp.process(input_path, video_blurred_path, fps=fps_str, orig_width=width, orig_height=height)
            else:
                # 3. Process chunks in parallel
                active_workers = max(1, min(workers, len(input_chunks)))
                update(f"Распознавание лиц и блюр (Потоков: {active_workers})...")
                t0 = time.time()
                processed_chunks_dir = os.path.join(job_dir, "processed_chunks")
                os.makedirs(processed_chunks_dir, exist_ok=True)
                
                manager = Manager()
                log_queue = manager.Queue()
                
                def log_listener():
                    while True:
                        try:
                            msg = log_queue.get()
                            if msg == "DONE":
                                break
                            update(msg)
                        except Exception:
                            break

                listener_thread = threading.Thread(target=log_listener)
                listener_thread.start()

                futures_map = {}
                with ProcessPoolExecutor(max_workers=active_workers, initializer=init_worker) as executor:
                    for idx, chunk in enumerate(input_chunks):
                        out_chunk = os.path.join(processed_chunks_dir, f"out_{idx:04d}.mp4")
                        f = executor.submit(process_chunk, chunk, out_chunk, fps_str, width, height, log_queue)
                        futures_map[f] = out_chunk
                
                    output_chunks = []
                    completed_count = 0
                    total_chunks = len(futures_map)
                    for future in as_completed(futures_map):
                        out_chunk = futures_map[future]
                        _, chunk_duration = future.result() # wait for completion
                        completed_count += 1
                        update(f"[Profile] Finished: {os.path.basename(out_chunk)} ({completed_count}/{total_chunks}) in {chunk_duration:.2f}s")
                        output_chunks.append(out_chunk)
                    
                log_queue.put("DONE")
                listener_thread.join()
                
                update(f"[Profile] All chunks processed in {time.time() - t0:.2f}s")
                    
                # 4. Concat processed chunks
                update("Склейка обработанных кусков...")
                t0 = time.time()
                list_file_path = os.path.join(job_dir, "concat_list.txt")
                with open(list_file_path, "w") as f:
                    for out_chunk in sorted(output_chunks):
                        safe_path = out_chunk.replace('\\', '/')
                        f.write(f"file '{safe_path}'\n")
                        
                video_blurred_path = os.path.join(job_dir, "video_blurred.mp4")
                FFmpegUtils.concat_videos(list_file_path, video_blurred_path)
                update(f"[Profile] Concat chunks finished in {time.time() - t0:.2f}s")
            
            # 5. Wait for Audio
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
            
            # 6. Mux final result
            update("Сборка финального видео...")
            t0 = time.time()
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            FFmpegUtils.mux(video_blurred_path, final_audio_path, output_path)
            update(f"[Profile] Mux finished in {time.time() - t0:.2f}s")

            # 7. Local Auto-Transcription with Speaker Diarization (if enabled)
            transcript_ready = False
            if transcribe and transcript_path:
                from src.transcriber import Transcriber, format_transcript_txt

                if parallel_gpu_future is not None:
                    if not parallel_gpu_future.done():
                        update("Ожидание завершения GPU-транскрибации...")
                    raw_wav_ready, transcript_ready = parallel_gpu_future.result()

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
                elif parallel_gpu_future is None:
                    update("Запуск локальной автотранскрибации по ролям...")
                    Transcriber.transcribe_audio_file(
                        wav_path=audio_16k_path,
                        output_txt_path=transcript_path,
                        asr_model=asr_model,
                        speaker_count=speaker_count,
                        threads=workers,
                        compute_device=requested_device,
                        status_cb=update,
                    )
                    transcript_ready = os.path.isfile(transcript_path)
        finally:
            audio_executor.shutdown(wait=True)
        
        update(f"Success! Output saved to: {output_path}. Total time: {time.time() - total_start_time:.2f}s")
        return {
            "video_ready": os.path.isfile(output_path),
            "transcript_ready": transcript_ready,
            "transcript_path": transcript_path if transcript_ready else None,
        }
