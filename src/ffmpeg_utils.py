import os
from fractions import Fraction
import ffmpeg
from src.cancellation import CancellationToken, TaskCancelledError
from src.config import MAX_PROCESSING_FPS, OUTPUT_VIDEO_CRF, OUTPUT_VIDEO_PRESET

class FFmpegUtils:
    @staticmethod
    def _run_stream(stream_spec, cancel_token: CancellationToken | None = None):
        if cancel_token is None:
            stream_spec.run(overwrite_output=True, quiet=True)
            return

        cancel_token.check()
        proc = ffmpeg.run_async(stream_spec, overwrite_output=True, quiet=True)
        cancel_token.register_proc(proc)
        try:
            out, err = proc.communicate()
        finally:
            cancel_token.unregister_proc(proc)
        cancel_token.check()
        if proc.returncode != 0:
            raise ffmpeg.Error("ffmpeg", out or b"", err or b"")

    @staticmethod
    def _normalize_fps(video_stream) -> str:
        def parse_rate(rate):
            if not rate or rate == '0/0':
                return None
            try:
                value = Fraction(rate)
            except (ValueError, ZeroDivisionError):
                return None
            return value if value > 0 else None

        # Prefer the nominal stream rate for processing. On VFR files
        # avg_frame_rate can be a long, low precision fraction that reflects
        # the whole file average rather than a useful CFR target.
        rates = [
            video_stream.get('r_frame_rate'),
            video_stream.get('avg_frame_rate'),
        ]
        
        selected_rate = None
        for rate in rates:
            selected_rate = parse_rate(rate)
            if selected_rate is not None:
                break
                
        if selected_rate is None:
            selected_rate = Fraction(30, 1)
            
        if MAX_PROCESSING_FPS:
            fps_limit = Fraction(str(MAX_PROCESSING_FPS)).limit_denominator(1001)
            if selected_rate > fps_limit:
                selected_rate = fps_limit
            
        selected_rate = selected_rate.limit_denominator(1001)
        return f"{selected_rate.numerator}/{selected_rate.denominator}"

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
    def mux(
        video_path: str,
        audio_path: str | None,
        output_path: str,
        cancel_token: CancellationToken | None = None,
        video_is_h264: bool = False,
    ):
        is_h264 = video_is_h264
        if not is_h264:
            try:
                probe = ffmpeg.probe(video_path)
                video_stream = next((s for s in probe.get('streams', []) if s['codec_type'] == 'video'), None)
                if video_stream and video_stream.get('codec_name') == 'h264':
                    is_h264 = True
            except Exception:
                pass

        v = ffmpeg.input(video_path)

        if is_h264:
            video_args = {
                'c:v': 'copy',
                'movflags': '+faststart',
            }
        else:
            video_args = {
                'c:v': 'libx264',
                'preset': OUTPUT_VIDEO_PRESET,
                'crf': str(OUTPUT_VIDEO_CRF),
                'movflags': '+faststart',
            }

        if audio_path and os.path.exists(audio_path):
            a = ffmpeg.input(audio_path)
            if audio_path.lower().endswith(('.m4a', '.aac')):
                audio_args = {'acodec': 'copy'}
            else:
                audio_args = {'acodec': 'aac', 'audio_bitrate': '128k'}
            spec = ffmpeg.output(v.video, a.audio, output_path, **video_args, **audio_args)
        else:
            spec = ffmpeg.output(v.video, output_path, **video_args)
        FFmpegUtils._run_stream(spec, cancel_token=cancel_token)

    @staticmethod
    def concat_and_mux(
        input_list_path: str,
        audio_path: str | None,
        output_path: str,
        cancel_token: CancellationToken | None = None,
    ):
        """
        Concatenates H.264 video chunks and muxes audio in a single stream-copy
        pass, avoiding intermediate multi-GB disk writes and sequential AAC encoding.
        """
        v = ffmpeg.input(input_list_path, format='concat', safe=0)
        common_args = {'c:v': 'copy', 'movflags': '+faststart'}

        if audio_path and os.path.exists(audio_path):
            a = ffmpeg.input(audio_path)
            if audio_path.lower().endswith(('.m4a', '.aac')):
                audio_args = {'c:a': 'copy'}
            else:
                audio_args = {'c:a': 'aac', 'audio_bitrate': '128k'}
            spec = ffmpeg.output(v.video, a.audio, output_path, **common_args, **audio_args)
        else:
            spec = ffmpeg.output(v.video, output_path, **common_args)
        FFmpegUtils._run_stream(spec, cancel_token=cancel_token)

    @staticmethod
    def split_video(
        input_path: str,
        output_pattern: str,
        chunk_time: int,
        cancel_token: CancellationToken | None = None,
    ):
        """Splits video into segments of chunk_time seconds."""
        # Using segment format in ffmpeg (stream copy is instant)
        spec = (
            ffmpeg
            .input(input_path)
            .output(output_pattern, map='0:v:0', c='copy', an=None, f='segment', segment_time=chunk_time, reset_timestamps=1)
        )
        FFmpegUtils._run_stream(spec, cancel_token=cancel_token)

    @staticmethod
    def concat_videos(
        input_list_path: str,
        output_path: str,
        cancel_token: CancellationToken | None = None,
    ):
        """Concatenates videos listed in a text file."""
        # The input list file must follow ffmpeg concat format: file 'path/to/file.mp4'
        spec = (
            ffmpeg
            .input(input_list_path, format='concat', safe=0)
            .output(output_path, c='copy')
        )
        FFmpegUtils._run_stream(spec, cancel_token=cancel_token)


    @staticmethod
    def extract_transcription_audio(
        input_path: str,
        output_wav_path: str,
        cancel_token: CancellationToken | None = None,
    ) -> bool:
        """Extracts original 16 kHz mono 16-bit PCM WAV for speaker diarization and ASR."""
        if not os.path.exists(input_path):
            return False
        try:
            spec = (
                ffmpeg
                .input(input_path)
                .output(
                    output_wav_path,
                    map='0:a:0',
                    vn=None,
                    ac=1,
                    ar=16000,
                    acodec='pcm_s16le',
                )
            )
            FFmpegUtils._run_stream(spec, cancel_token=cancel_token)
            return os.path.isfile(output_wav_path) and os.path.getsize(output_wav_path) > 44
        except TaskCancelledError:
            raise
        except ffmpeg.Error as e:
            err_msg = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else str(e)
            print(f"Error extracting transcription audio with FFmpeg: {err_msg}")
            return False

