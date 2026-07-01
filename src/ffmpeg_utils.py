import os
from fractions import Fraction
import ffmpeg
from src.config import MAX_PROCESSING_FPS

class FFmpegUtils:
    @staticmethod
    def _normalize_fps(video_stream) -> str:
        rates = [
            video_stream.get('avg_frame_rate'),
            video_stream.get('r_frame_rate'),
        ]
        
        selected_rate = None
        selected_value = 0.0
        for rate in rates:
            if not rate or rate == '0/0':
                continue
            try:
                value = float(Fraction(rate))
            except (ValueError, ZeroDivisionError):
                continue
            if value > 0:
                selected_rate = rate
                selected_value = value
                break
                
        if selected_rate is None:
            return '30/1'
            
        if MAX_PROCESSING_FPS and selected_value > MAX_PROCESSING_FPS:
            return f"{int(MAX_PROCESSING_FPS)}/1"
            
        return selected_rate

    @staticmethod
    def get_video_metadata(input_path: str):
        width, height, fps = None, None, '30/1'
        duration = 0.0
        has_audio = False
        sample_rate = 44100
        
        try:
            probe = ffmpeg.probe(input_path)
            
            if 'format' in probe and 'duration' in probe['format']:
                duration = float(probe['format']['duration'])
                
            for stream in probe.get('streams', []):
                if stream['codec_type'] == 'video' and width is None:
                    width = int(stream.get('width', 0))
                    height = int(stream.get('height', 0))
                    fps = FFmpegUtils._normalize_fps(stream)
                elif stream['codec_type'] == 'audio' and not has_audio:
                    has_audio = True
                    sample_rate = int(stream.get('sample_rate', 44100))
                    
        except ffmpeg.Error:
            pass
            
        return width, height, fps, duration, has_audio, sample_rate

    @staticmethod
    def mux(video_path: str, audio_path: str, output_path: str):
        is_h264 = False
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((s for s in probe.get('streams', []) if s['codec_type'] == 'video'), None)
            if video_stream and video_stream.get('codec_name') == 'h264':
                is_h264 = True
        except:
            pass

        v = ffmpeg.input(video_path)
        
        if is_h264:
            video_args = {
                'c:v': 'copy',
                'movflags': '+faststart'
            }
        else:
            video_args = {
                'c:v': 'libx264',
                'preset': 'ultrafast',
                'crf': '23',
                'movflags': '+faststart'
            }
        
        if audio_path and os.path.exists(audio_path):
            a = ffmpeg.input(audio_path)
            ffmpeg.output(v, a, output_path, **video_args, acodec='aac', audio_bitrate='128k').run(overwrite_output=True, quiet=True)
        else:
            ffmpeg.output(v, output_path, **video_args).run(overwrite_output=True, quiet=True)

    @staticmethod
    def split_video(input_path: str, output_pattern: str, chunk_time: int):
        """Splits video into segments of chunk_time seconds."""
        # Using segment format in ffmpeg (stream copy is instant)
        (
            ffmpeg
            .input(input_path)
            .output(output_pattern, map='0:v:0', c='copy', an=None, f='segment', segment_time=chunk_time, reset_timestamps=1)
            .run(overwrite_output=True, quiet=True)
        )
        
    @staticmethod
    def concat_videos(input_list_path: str, output_path: str):
        """Concatenates videos listed in a text file."""
        # The input list file must follow ffmpeg concat format: file 'path/to/file.mp4'
        (
            ffmpeg
            .input(input_list_path, format='concat', safe=0)
            .output(output_path, c='copy')
            .run(overwrite_output=True, quiet=True)
        )
