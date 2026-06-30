import os
import glob
from concurrent.futures import ProcessPoolExecutor
from src.io_manager import IOManager
from src.ffmpeg_utils import FFmpegUtils
from src.audio_processor import AudioProcessor
from src.video_processor import VideoProcessor
from src.config import MAX_WORKERS, CHUNK_DURATION_SECONDS

def process_chunk(input_chunk: str, output_chunk: str) -> bool:
    vp = VideoProcessor()
    return vp.process(input_chunk, output_chunk)

class Orchestrator:
    @staticmethod
    def process_video_job(input_path: str, output_path: str, job_dir: str, status_cb=None):
        def update(msg):
            print(msg)
            if status_cb:
                status_cb(msg)
                
        update(f"Starting processing for: {input_path}")
        
        video_mute_path = os.path.join(job_dir, "video_mute.mp4")
        audio_raw_path = os.path.join(job_dir, "audio_raw.wav")
        audio_pitched_path = os.path.join(job_dir, "audio_pitched.wav")
        
        # 1. Demux
        update("Извлечение аудио и видео...")
        has_audio = FFmpegUtils.demux(input_path, video_mute_path, audio_raw_path)
        
        # 2. Process Audio
        if has_audio:
            update("Анонимизация голоса...")
            audio_success = AudioProcessor.process(audio_raw_path, audio_pitched_path)
            if not audio_success:
                audio_pitched_path = audio_raw_path
        
        # 3. Split video into chunks
        update("Разделение видео для параллельной обработки...")
        chunks_dir = os.path.join(job_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        chunk_pattern = os.path.join(chunks_dir, "chunk_%04d.mp4")
        FFmpegUtils.split_video(video_mute_path, chunk_pattern, CHUNK_DURATION_SECONDS)
        
        input_chunks = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.mp4")))
        
        if not input_chunks:
            # Fallback
            update("Распознавание лиц и блюр...")
            video_blurred_path = os.path.join(job_dir, "video_blurred.mp4")
            vp = VideoProcessor()
            vp.process(video_mute_path, video_blurred_path)
        else:
            # 4. Process chunks in parallel
            update(f"Распознавание лиц и блюр (Потоков: {MAX_WORKERS})...")
            processed_chunks_dir = os.path.join(job_dir, "processed_chunks")
            os.makedirs(processed_chunks_dir, exist_ok=True)
            
            futures = []
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for idx, chunk in enumerate(input_chunks):
                    out_chunk = os.path.join(processed_chunks_dir, f"out_{idx:04d}.mp4")
                    futures.append((executor.submit(process_chunk, chunk, out_chunk), out_chunk))
            
            output_chunks = []
            for future, out_chunk in futures:
                future.result() # wait for completion
                output_chunks.append(out_chunk)
                
            # 5. Concat processed chunks
            update("Склейка обработанных кусков...")
            list_file_path = os.path.join(job_dir, "concat_list.txt")
            with open(list_file_path, "w") as f:
                for out_chunk in sorted(output_chunks):
                    safe_path = out_chunk.replace('\\', '/')
                    f.write(f"file '{safe_path}'\n")
                    
            video_blurred_path = os.path.join(job_dir, "video_blurred.mp4")
            FFmpegUtils.concat_videos(list_file_path, video_blurred_path)
        
        # 6. Mux final result
        update("Сборка финального видео (Сжатие в H.264)...")
        final_audio_path = audio_pitched_path if has_audio else None
        FFmpegUtils.mux(video_blurred_path, final_audio_path, output_path)
        
        update(f"Success! Output saved to: {output_path}")
