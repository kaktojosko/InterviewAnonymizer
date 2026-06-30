import os
import ffmpeg

class FFmpegUtils:
    @staticmethod
    def get_video_info(input_path: str):
        try:
            probe = ffmpeg.probe(input_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream:
                width = int(video_stream['width'])
                height = int(video_stream['height'])
                return width, height
        except ffmpeg.Error:
            pass
        return None, None

    @staticmethod
    def demux(input_path: str, output_video_path: str, output_audio_path: str) -> bool:
        v = ffmpeg.input(input_path)
        
        # Optimization: Check if we actually need to downscale
        width, height = FFmpegUtils.get_video_info(input_path)
        
        # If it's 1080p or smaller, DO NOT RE-ENCODE! Just copy the stream (takes ~1 second)
        if height and height <= 1080:
            ffmpeg.output(v.video, output_video_path, c='copy').run(overwrite_output=True, quiet=True)
        else:
            # If it's 4K+, scale it but use 'ultrafast' since it's just an intermediate file
            ffmpeg.output(v.video, output_video_path, vf="scale=-2:'min(1080,ih)'", vcodec='libx264', preset='ultrafast', crf=18).run(overwrite_output=True, quiet=True)
            
        try:
            # Extract audio quickly (no heavy compression needed here)
            ffmpeg.output(v.audio, output_audio_path, acodec='pcm_s16le').run(overwrite_output=True, quiet=True)
            return True
        except ffmpeg.Error:
            return False

    @staticmethod
    def mux(video_path: str, audio_path: str, output_path: str):
        v = ffmpeg.input(video_path)
        
        # Оптимизация H.264:
        # preset='faster' - отличный баланс между скоростью и размером
        # crf=23 - визуально без потерь (visually lossless)
        # faststart - оптимизация для веб-плееров
        video_args = {
            'vcodec': 'libx264',
            'preset': 'faster',
            'crf': 23,
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
            .output(output_pattern, c='copy', f='segment', segment_time=chunk_time, reset_timestamps=1)
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
