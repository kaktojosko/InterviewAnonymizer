import os
import time
import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from src.io_manager import IOManager
from src.ffmpeg_utils import FFmpegUtils
from src.audio_processor import AudioProcessor
from src.video_processor import VideoProcessor
from src.config import MAX_WORKERS, CHUNK_DURATION_SECONDS

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
    def process_video_job(input_path: str, output_path: str, job_dir: str, status_cb=None):
        def update(msg):
            print(msg)
            if status_cb:
                status_cb(msg)
                
        update(f"Starting processing for: {input_path}")
        total_start_time = time.time()
        
        audio_pitched_path = os.path.join(job_dir, "audio_pitched.wav")
        
        # 1. Probe (Metadata)
        update("Анализ видеофайла...")
        t0 = time.time()
        width, height, fps_str, duration, has_audio, sample_rate = FFmpegUtils.get_video_metadata(input_path)
        update(f"[Profile] Probe finished in {time.time() - t0:.2f}s")
        
        # 2. Parallel Audio & Split Video
        audio_future = None
        audio_executor = ThreadPoolExecutor(max_workers=1)
        if has_audio:
            update("Запуск обработки аудио (в фоне)...")
            audio_future = audio_executor.submit(AudioProcessor.process, input_path, audio_pitched_path, sample_rate)
            
        update("Разделение видео для параллельной обработки...")
        t0 = time.time()
        
        if duration > 0:
            adaptive_chunk_time = max(30, min(90, int(duration / MAX_WORKERS)))
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
            update(f"Распознавание лиц и блюр (Потоков: {MAX_WORKERS})...")
            t0 = time.time()
            processed_chunks_dir = os.path.join(job_dir, "processed_chunks")
            os.makedirs(processed_chunks_dir, exist_ok=True)
            
            from concurrent.futures import as_completed
            from multiprocessing import Manager
            import threading
            
            manager = Manager()
            log_queue = manager.Queue()
            
            def log_listener():
                while True:
                    try:
                        msg = log_queue.get()
                        if msg == "DONE":
                            break
                        update(msg)
                    except:
                        break

            listener_thread = threading.Thread(target=log_listener)
            listener_thread.start()

            futures_map = {}
            with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
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
        if has_audio and audio_future:
            update("Ожидание завершения обработки аудио...")
            t0 = time.time()
            audio_success = audio_future.result()
            update(f"[Profile] Audio wait finished in {time.time() - t0:.2f}s")
            if audio_success:
                final_audio_path = audio_pitched_path
            audio_executor.shutdown()
        
        # 6. Mux final result
        update("Сборка финального видео...")
        t0 = time.time()
        FFmpegUtils.mux(video_blurred_path, final_audio_path, output_path)
        update(f"[Profile] Mux finished in {time.time() - t0:.2f}s")
        
        update(f"Success! Output saved to: {output_path}. Total time: {time.time() - total_start_time:.2f}s")
