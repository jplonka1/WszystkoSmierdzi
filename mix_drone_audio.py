import os
import random
from pydub import AudioSegment

def mix_audio_with_fade(background_path, foreground_path, output_folder, output_filename=None):
    # Load audio files
    background = AudioSegment.from_wav(background_path)
    foreground = AudioSegment.from_wav(foreground_path)
    
    # Check lengths
    if len(foreground) > len(background):
        raise ValueError("Foreground is longer than background")
    
    # Apply fade-in and randomized gain to foreground
    fade_duration = len(foreground)
    fade_end_db = random.uniform(-3, 0)  # volume from ~70% to 100%
    foreground = foreground.fade_in(fade_duration).apply_gain(fade_end_db)
    
    # Pick random insertion time
    max_insert_position = len(background) - len(foreground)
    insert_position = random.randint(0, max_insert_position)
    
    # Overlay the foreground onto the background
    mixed = background.overlay(foreground, position=insert_position)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create output filename if not provided
    if output_filename is None:
        # Example: output_12345.wav
        output_filename = f"output_{random.randint(10000,99999)}.wav"
        
    output_path = os.path.join(output_folder, output_filename)
    
    # Export the final clip
    mixed.export(output_path, format="wav")
    
    return output_path
