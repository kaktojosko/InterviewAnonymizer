import os
import sys
import argparse
from src.io_manager import IOManager
from src.ffmpeg_utils import FFmpegUtils
from src.audio_processor import AudioProcessor
from src.video_processor import VideoProcessor

def process_video(input_path: str, output_path: str):
    print(f"Starting processing for: {input_path}")
    
    io_manager = IOManager()
    job_dir = io_manager.create_job_dir()
    
    try:
        # Paths
        video_mute_path = os.path.join(job_dir, "video_mute.mp4")
        audio_raw_path = os.path.join(job_dir, "audio_raw.wav")
        video_blurred_path = os.path.join(job_dir, "video_blurred.mp4")
        audio_pitched_path = os.path.join(job_dir, "audio_pitched.wav")
        
        # 1. Demux
        print("Demuxing video and audio...")
        has_audio = FFmpegUtils.demux(input_path, video_mute_path, audio_raw_path)
        
        # 2. Process Audio
        if has_audio:
            print("Processing audio (Pitch Shifting)...")
            audio_success = AudioProcessor.process(audio_raw_path, audio_pitched_path)
            if not audio_success:
                print("Audio processing failed, continuing with original audio.")
                audio_pitched_path = audio_raw_path
        else:
            print("No audio stream found. Skipping audio processing.")
            
        # 3. Process Video
        print("Processing video (Face Detection & Blurring)... This might take a while.")
        vp = VideoProcessor()
        video_success = vp.process(video_mute_path, video_blurred_path)
        if not video_success:
            print("Video processing failed!")
            return
            
        # 4. Mux
        print("Muxing video and audio...")
        final_audio_path = audio_pitched_path if has_audio else None
        FFmpegUtils.mux(video_blurred_path, final_audio_path, output_path)
        
        print(f"Success! Output saved to: {output_path}")
        
    finally:
        # Cleanup
        print("Cleaning up temporary files...")
        io_manager.cleanup_job_dir(job_dir)

def main():
    parser = argparse.ArgumentParser(description="Interview Anonymizer (Blur faces and pitch shift audio)")
    parser.add_argument("-i", "--input", required=True, help="Path to input video")
    parser.add_argument("-o", "--output", required=True, help="Path to output video")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
        
    process_video(args.input, args.output)

if __name__ == "__main__":
    main()
