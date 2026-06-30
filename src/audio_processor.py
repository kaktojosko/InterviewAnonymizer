import os
import ffmpeg
from src.config import PITCH_SHIFT_SEMITONES

class AudioProcessor:
    @staticmethod
    def process(input_audio_path: str, output_audio_path: str) -> bool:
        """
        Applies pitch shifting to the audio file using FFmpeg's native filters.
        Extremely fast (does not load full file into RAM).
        """
        if not os.path.exists(input_audio_path):
            return False
            
        try:
            # Probe sample rate
            probe = ffmpeg.probe(input_audio_path)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            sample_rate = 44100
            if audio_stream and 'sample_rate' in audio_stream:
                sample_rate = int(audio_stream['sample_rate'])
                
            # Calculate pitch shift multipliers
            rate_multiplier = 2 ** (PITCH_SHIFT_SEMITONES / 12.0)
            
            new_rate = int(sample_rate * rate_multiplier)
            atempo = 1.0 / rate_multiplier
            
            # asetrate changes sample rate (pitch + speed)
            # aresample restores standard sample rate
            # atempo fixes the speed
            af_filter = f"asetrate={new_rate},aresample={sample_rate},atempo={atempo:.4f}"
            
            (
                ffmpeg
                .input(input_audio_path)
                .output(output_audio_path, af=af_filter)
                .run(overwrite_output=True, quiet=True)
            )
            return True
        except ffmpeg.Error as e:
            err_msg = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else str(e)
            print(f"Error processing audio with FFmpeg: {err_msg}")
            return False
