import os
import soundfile as sf
from pedalboard import Pedalboard, PitchShift
from src.config import PITCH_SHIFT_SEMITONES

class AudioProcessor:
    @staticmethod
    def process(input_audio_path: str, output_audio_path: str) -> bool:
        """
        Applies pitch shifting to the audio file using Pedalboard (Spotify) 
        to anonymize the voice with high quality.
        Returns True if successful.
        """
        if not os.path.exists(input_audio_path):
            return False
            
        try:
            # Read audio data
            # Pedalboard works well with numpy arrays provided by soundfile
            audio_data, sample_rate = sf.read(input_audio_path)
            
            # soundfile returns (frames, channels) or (frames,) for mono.
            # Pedalboard expects (channels, frames).
            if len(audio_data.shape) == 1:
                # Mono
                audio_data = audio_data.reshape(1, -1)
            else:
                # Stereo or multi-channel
                audio_data = audio_data.T
                
            # Create Pedalboard with PitchShift
            board = Pedalboard([
                PitchShift(semitones=PITCH_SHIFT_SEMITONES)
            ])
            
            # Process audio
            processed_audio = board(audio_data, sample_rate)
            
            # Convert back to (frames, channels) for saving
            processed_audio = processed_audio.T
            
            # Save the result
            sf.write(output_audio_path, processed_audio, sample_rate)
            
            return True
        except Exception as e:
            print(f"Error processing audio with Pedalboard: {e}")
            return False
