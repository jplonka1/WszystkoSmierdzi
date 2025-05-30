import os
import random
from pydub import AudioSegment

### This script mixes foreground audio files with unique background audio files,
### applying a fade-in and fade-out effect to the foreground audio.

def mix_audio_with_fade(background_path, foreground_path, output_folder, output_filename=None):
    # Load audio files
    background = AudioSegment.from_wav(background_path)
    foreground = AudioSegment.from_wav(foreground_path)
    
    # Check lengths
    if len(foreground) > len(background):
        print(f"Skipping: Foreground '{foreground_path}' is longer than background '{background_path}'")
        return None
    
    # Apply fade-in and fade-out, and randomized gain to foreground
    fade_duration = len(foreground)
    fade_end_db = random.uniform(-1, 0)  # volume from ~70% to 100%
    foreground = foreground.fade_in(fade_duration).fade_out(fade_duration).apply_gain(fade_end_db)
    
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

### Function to batch mix foregrounds with unique backgrounds (for folder structure).

def batch_mix_foregrounds_with_unique_backgrounds(foreground_folder, background_folder, output_folder):
    # List files
    foreground_files = sorted([
        os.path.join(foreground_folder, f) 
        for f in os.listdir(foreground_folder) if f.lower().endswith(".wav")
    ])
    background_files = [
        os.path.join(background_folder, f) 
        for f in os.listdir(background_folder) if f.lower().endswith(".wav")
    ]
    
    if len(background_files) < len(foreground_files):
        raise ValueError("Not enough background files to match each foreground uniquely.")
    
    # Shuffle backgrounds so selection is random
    random.shuffle(background_files)
    
    used_backgrounds = set()
    output_paths = []
    
    for fg_path in foreground_files:
        # Pick a random unused background
        bg_path = background_files.pop()  # pop from shuffled list - guarantees unique and random
        
        # Compose output filename for clarity
        fg_name = os.path.splitext(os.path.basename(fg_path))[0]
        bg_name = os.path.splitext(os.path.basename(bg_path))[0]
        output_filename = f"{fg_name}_with_{bg_name}.wav"
        
        # Mix and save
        output_path = mix_audio_with_fade(bg_path, fg_path, output_folder, output_filename)
        output_paths.append(output_path)
        print(f"Mixed {fg_name} with {bg_name} → saved as {output_filename}")
    
    return output_paths

batch_mix_foregrounds_with_unique_backgrounds("Drone", "Background", "Drone_mixed")
