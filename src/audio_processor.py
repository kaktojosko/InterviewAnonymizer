import os
import ffmpeg
from src.cancellation import CancellationToken, TaskCancelledError
from src.config import PITCH_SHIFT_SEMITONES
from src.ffmpeg_utils import FFmpegUtils

class AudioProcessor:
    @staticmethod
    def process(
        input_audio_path: str,
        output_audio_path: str,
        sample_rate: int = 44100,
        cancel_token: CancellationToken | None = None,
        num_audio_streams: int = 1,
        mix_all_streams: bool = False,
    ) -> bool:
        """
        Applies pitch shifting to the audio file using FFmpeg's native filters.
        Extremely fast (does not load full file into RAM). When mix_all_streams is
        True on multi-track OBS recordings, mixes all audio tracks before pitch shifting.
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
            valid_aac_rates = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
            out_rate = sample_rate if sample_rate in valid_aac_rates else 44100
            af_filter = f"asetrate={new_rate},aresample={out_rate},atempo={atempo:.4f}"

            out_kwargs = {"vn": None}
            if output_audio_path.lower().endswith((".m4a", ".aac", ".mp4")):
                out_kwargs["acodec"] = "aac"
                out_kwargs["audio_bitrate"] = "128k"

            inp = ffmpeg.input(input_audio_path)
            if mix_all_streams and int(num_audio_streams) > 1:
                n = int(num_audio_streams)
                audio_inputs = [inp[f"a:{i}"] for i in range(n)]
                mixed = (
                    ffmpeg.filter(
                        audio_inputs,
                        "amix",
                        inputs=n,
                        duration="longest",
                        dropout_transition=0,
                        normalize=0,
                    )
                    .filter("asetrate", new_rate)
                    .filter("aresample", out_rate)
                    .filter("atempo", f"{atempo:.4f}")
                )
                spec = ffmpeg.output(mixed, output_audio_path, **out_kwargs)
            else:
                out_kwargs["map"] = "0:a:0"
                out_kwargs["af"] = af_filter
                spec = inp.output(output_audio_path, **out_kwargs)

            FFmpegUtils._run_stream(spec, cancel_token=cancel_token)
            return True
        except TaskCancelledError:
            raise
        except ffmpeg.Error as e:
            err_msg = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else str(e)
            print(f"Error processing audio with FFmpeg: {err_msg}")
            return False

