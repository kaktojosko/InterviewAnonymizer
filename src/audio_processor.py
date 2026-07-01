import os
import ffmpeg
from src.config import PITCH_SHIFT_SEMITONES

class AudioProcessor:
    @staticmethod
    def process(input_audio_path: str, output_audio_path: str, sample_rate: int = 44100) -> bool:
        """
        Applies pitch shifting to the audio file using FFmpeg's native filters.
        Extremely fast (does not load full file into RAM).
        """
        if not os.path.exists(input_audio_path):
            return False
            
        try:
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
                .output(output_audio_path, map='0:a:0', af=af_filter)
                .run(overwrite_output=True, quiet=True)
            )
            return True
        except ffmpeg.Error as e:
            err_msg = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else str(e)
            print(f"Error processing audio with FFmpeg: {err_msg}")
            return False
