### PRZESTARZAŁA FUNKCJA, WSZYSTKO W AUGMENTATION.PY
####################
####################
####################
import numpy as np

def apply_fade(audio, sr, fade_type='in_out'):
    """Apply fade-in and/or fade-out to the entire audio."""
    length = len(audio)
    fade_samples = np.linspace(0, 1, length)
    
    if fade_type == 'in':
        fade = fade_samples
    elif fade_type == 'out':
        fade = fade_samples[::-1]
    elif fade_type == 'in_out':
        fade_in = np.linspace(0, 1, length // 2)
        fade_out = np.linspace(1, 0, length - len(fade_in))
        fade = np.concatenate((fade_in, fade_out))
    else:
        raise ValueError("fade_type must be 'in', 'out', or 'in_out'")
    
    return audio * fade

def insert_with_fade(background, snippet, sr, fade=True, gain_db_range=(-3, 0)):
    """Insert a snippet into the background with optional fading and random gain."""
    snippet = snippet.copy()
    if fade:
        snippet = apply_fade(snippet, sr, fade_type='in_out')
    
    # Random gain (in dB)
    gain_db = np.random.uniform(*gain_db_range)
    gain_lin = 10 ** (gain_db / 20)
    snippet *= gain_lin
    
    # Random position to insert
    max_start = len(background) - len(snippet)
    if max_start <= 0:
        return background  # skip insertion if snippet is longer than background
    start = np.random.randint(0, max_start)
    
    # Insert: mix snippet into background
    mixed = background.copy()
    mixed[start:start + len(snippet)] += snippet
    return mixed

def mix_librosa_audio(background, drone, noise, sr):
    """
    Mix drone and noise into the background with fade-in/out and random timing.
    """
    # Ensure all signals are the same length or shorter than background
    background = background.copy()
    
    if drone is not None and len(drone) > len(background):
        drone = drone[:len(background)]
    if len(noise) > len(background):
        noise = noise[:len(background)]

    # Mix drone and noise with background
    if drone is not None:
        background = insert_with_fade(background, drone, sr, fade=True, gain_db_range=(-4, 0))
    background = insert_with_fade(background, noise, sr, fade=True, gain_db_range=(-4, -1))

    # Normalize to prevent clipping
    peak = np.max(np.abs(background))
    if peak > 1:
        background /= peak

    return background