import numpy as np
import librosa
import torch
import soundfile as sf
import os
import random
from scipy.signal import butter, lfilter
from scipy.stats import truncnorm

### AUGMENTACJA

# -------- Utility for truncated normal --------
def truncated_normal(mean, std, lower, upper):
    return truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std).rvs()

# -------- Augmentation Functions --------
def amplitude_clipping(audio, clip_range=(0.75, 1.0)):
    clip_amp = truncated_normal(0.875, 0.05, *clip_range)
    max_val = np.max(np.abs(audio))
    return np.clip(audio, -clip_amp * max_val, clip_amp * max_val)

def volume_amplify(audio, amp_range=(0.5, 1.5)):
    gain = truncated_normal(1.0, 0.25, *amp_range)
    return audio * gain

def add_echo(audio, sr, delay_range=(0.02, 0.4), decay=0.5):
    delay_sec = truncated_normal(0.21, 0.1, *delay_range)
    delay_samples = int(delay_sec * sr)
    echo = np.zeros_like(audio)
    if delay_samples < len(audio):
        echo[delay_samples:] = audio[:-delay_samples] * decay
    return audio + echo

def partial_erase(audio, sr, erase_ratio=(0.1, 0.3), chunk_ratio_range=(0.01, 0.05)):
    """
    Erases multiple small chunks of the audio with total duration proportional to audio length.

    Parameters:
        audio (np.ndarray): Audio signal.
        sr (int): Sample rate.
        erase_ratio (tuple): Min and max fraction of audio to erase (e.g., 0.1 to 0.3 for 10%–30%).
        chunk_ratio_range (tuple): Min and max chunk length as fraction of total duration 
                                   (e.g., 0.01 to 0.05 for 1%–5%).

    Returns:
        np.ndarray: Modified audio.
    """
    audio = audio.copy()
    total_len = len(audio)
    total_duration = total_len / sr

    # Step 1: Choose total erase duration as a fraction of total audio duration
    total_erase_duration = truncated_normal(0.2, 0.1, *erase_ratio) * total_duration
    total_erase_samples = int(total_erase_duration * sr)

    erased = 0
    attempts = 0
    max_attempts = 1000  # Increased attempts for flexibility

    while erased < total_erase_samples and attempts < max_attempts:
        # Step 2: Sample a chunk length ratio, convert to samples
        chunk_ratio = truncated_normal(0.03, 0.01, *chunk_ratio_range)  # mean 3%, std 1%
        chunk_len = int(chunk_ratio * total_len)

        if chunk_len == 0 or chunk_len + erased > total_erase_samples or chunk_len >= total_len:
            attempts += 1
            continue

        # Step 3: Choose a random start location
        start = np.random.randint(0, total_len - chunk_len)

        # Step 4: Erase chunk by replacing with noise
        audio[start:start + chunk_len] = np.random.normal(0, 0.01, chunk_len)

        erased += chunk_len
        attempts += 1

    return audio

## PITCH SEEMS LIKE A STRONG EFFECT     
def pitch_shift(audio, sr, shift_range=(-5, 5)):
    shift_steps = truncated_normal(0, 2, *shift_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift_steps)

def add_noise(audio, noise_level=(0.005, 0.02)):
    noise_amp = truncated_normal(0.01, 0.005, *noise_level)
    return audio + np.random.normal(0, noise_amp, size=len(audio))

## THIS SEEMS LIKE MOSTLY A MUSIC AUGMENTATION, NOT FOR DRONES, NOISE ETC.
def hpss_separate(audio, min_harm=0.3, max_harm=0.9):
    harm, perc = librosa.effects.hpss(audio)
    harm_ratio = np.random.uniform(min_harm, max_harm)
    perc_ratio = 1.0 - harm_ratio
    augmented = harm_ratio * harm + perc_ratio * perc

    # Normalize
    max_val = np.max(np.abs(augmented))
    if max_val > 1.0:
        augmented = augmented / max_val

    return augmented

## USEFUL, IT ROLLS AUDIO (beginning is cut and put at the end)
def time_shift(audio, shift_max=0.2):
    shift_fraction = truncated_normal(0, 0.05, -shift_max, shift_max)
    shift = int(len(audio) * shift_fraction)
    return np.roll(audio, shift)

def tanh_distortion(audio, distortion_level=(1.0, 5.0)):
    strength = truncated_normal(2.0, 1.0, *distortion_level)
    return np.tanh(audio * strength)

def butter_filter(audio, sr, cutoff, btype='low', order=5):
    nyquist = 0.5 * sr
    norm_cutoff = np.array(cutoff) / nyquist
    b, a = butter(order, norm_cutoff, btype=btype, analog=False)
    return lfilter(b, a, audio)

def high_pass(audio, sr, cutoff_range=(100, 400)):
    # For drones, use a lower cutoff range to preserve low-frequency content
    cutoff = truncated_normal(200, 50, *cutoff_range)
    return butter_filter(audio, sr, cutoff, btype='high')

def band_pass(audio, sr, band_range=(50, 800)):
    # For drones, use a wider band to include more low-mid frequencies
    low = truncated_normal(200, 100, band_range[0], band_range[1])
    high = low + 600  # 600 Hz band width, suitable for drone harmonics
    return butter_filter(audio, sr, [low, high], btype='band')

# Applying both bands is unnatural, so we choose one randomly
def random_filter(audio, sr):
    choice = random.choice(['high', 'band'])
    if choice == 'high':
        return high_pass(audio, sr)
    else:
        return band_pass(audio, sr)

# -------- Augmentation Wrappers --------
def mild_augmentation(audio, sr):
    audio = amplitude_clipping(audio)
    audio = volume_amplify(audio)
    audio = partial_erase(audio)
    audio = add_noise(audio)
    return audio

def strong_augmentation(audio, sr):
    augmentations = [
        (amplitude_clipping, 0.8),
        (volume_amplify, 0.85),
        (add_echo, 0.5),
        (partial_erase, 1),
        (add_noise, 1),
        (pitch_shift, 0.5),
        (hpss_separate, 0.4),
        (time_shift, 1),
        (tanh_distortion, 0.3),
        (random_filter, 0.5),
    ]
    
    random.shuffle(augmentations)
    
    log = []
    audio_out = audio.copy()

    for func, prob in augmentations:
        if random.random() < prob:
            # Detect if function needs sr parameter
            try:
                # For functions requiring sr, get the parameters used explicitly for logging
                if func == amplitude_clipping:
                    clip_amp = truncated_normal(0.875, 0.05, 0.75, 1.0)
                    max_val = np.max(np.abs(audio_out))
                    audio_out = np.clip(audio_out, -clip_amp * max_val, clip_amp * max_val)
                    log.append(f"{func.__name__}(clip_amp={clip_amp:.4f})")

                elif func == volume_amplify:
                    gain = truncated_normal(1.0, 0.25, 0.5, 1.5)
                    audio_out = audio_out * gain
                    log.append(f"{func.__name__}(gain={gain:.4f})")

                elif func == add_echo:
                    delay_sec = truncated_normal(0.21, 0.1, 0.02, 0.4)
                    delay_samples = int(delay_sec * sr)
                    decay = 0.5
                    echo = np.zeros_like(audio_out)
                    if delay_samples < len(audio_out):
                        echo[delay_samples:] = audio_out[:-delay_samples] * decay
                    audio_out = audio_out + echo
                    log.append(f"{func.__name__}(delay_sec={delay_sec:.4f}, decay={decay})")

                elif func == partial_erase:
                    audio_out = partial_erase(audio_out, sr)
                    log.append(f"{func.__name__}(used default params)")

                elif func == add_noise:
                    noise_amp = truncated_normal(0.01, 0.005, 0.005, 0.02)
                    audio_out = audio_out + np.random.normal(0, noise_amp, size=len(audio_out))
                    log.append(f"{func.__name__}(noise_amp={noise_amp:.5f})")

                elif func == pitch_shift:
                    shift_steps = truncated_normal(0, 2, -5, 5)
                    audio_out = librosa.effects.pitch_shift(audio_out, sr=sr, n_steps=shift_steps)
                    log.append(f"{func.__name__}(shift_steps={shift_steps:.4f})")

                elif func == hpss_separate:
                    min_harm = 0.3
                    max_harm = 0.9
                    harm, perc = librosa.effects.hpss(audio_out)
                    harm_ratio = np.random.uniform(min_harm, max_harm)
                    perc_ratio = 1.0 - harm_ratio
                    augmented = harm_ratio * harm + perc_ratio * perc
                    max_val = np.max(np.abs(augmented))
                    if max_val > 1.0:
                        augmented = augmented / max_val
                    audio_out = augmented
                    log.append(f"{func.__name__}(harm_ratio={harm_ratio:.4f}, perc_ratio={perc_ratio:.4f})")

                elif func == time_shift:
                    shift_max = 0.2
                    shift_fraction = truncated_normal(0, 0.05, -shift_max, shift_max)
                    shift = int(len(audio_out) * shift_fraction)
                    audio_out = np.roll(audio_out, shift)
                    log.append(f"{func.__name__}(shift_fraction={shift_fraction:.4f}, shift_samples={shift})")

                elif func == tanh_distortion:
                    strength = truncated_normal(2.0, 1.0, 1.0, 5.0)
                    audio_out = np.tanh(audio_out * strength)
                    log.append(f"{func.__name__}(strength={strength:.4f})")

                elif func == random_filter:
                    choice = random.choice(['high', 'band'])
                    if choice == 'high':
                        cutoff = truncated_normal(200, 50, 100, 400)
                        audio_out = butter_filter(audio_out, sr, cutoff, btype='high')
                        log.append(f"{func.__name__}(choice='high', cutoff={cutoff:.2f})")
                    else:
                        low = truncated_normal(200, 100, 50, 800)
                        high = low + 600
                        audio_out = butter_filter(audio_out, sr, [low, high], btype='band')
                        log.append(f"{func.__name__}(choice='band', low={low:.2f}, high={high:.2f})")

            except Exception as e:
                # fallback, just try calling without sr if something goes wrong
                try:
                    audio_out = func(audio_out)
                    log.append(f"{func.__name__}()")
                except:
                    print(f"Skipping {func.__name__} due to error: {e}")

    return audio_out, log

# -------- Main Augmentation Logic --------
def augment_and_mix(background, drone, noise, sr, save_path=None):
    # Mild augmentation for background
    background_aug = mild_augmentation(background.copy(), sr)
    
    # Strong augmentation for noise
    noise_aug = strong_augmentation(noise.copy(), sr)
    label = 0

    # Drone might be None
    if drone is not None:
        drone_aug = strong_augmentation(drone.copy(), sr)
        label = 1
    else:
        drone_aug = None

    # Mix them using the provided mixing function
    mixed = mix_librosa_audio(background_aug, drone_aug, noise_aug, sr)
    mixed_tensor = torch.from_numpy(mixed).float()

    if save_path:
        filename = os.path.splitext(os.path.basename(save_path))[0]
        labelled_path = f"{filename}_label{label}.wav"
        sf.write(labelled_path, mixed, sr)
        print(f"[DEBUG] Saved: {labelled_path}")

    return mixed_tensor, label


###  MIXOWANIE

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


### EXAMPLE

#background, sr = librosa.load("001_0.wav", sr=None)
#drone = librosa.load("B_S2_D1_067-bebop_001_.wav", sr=None)[0] if np.random.rand() < 0.5 else None
#noise, _ = librosa.load("1-11687-A-472.wav", sr=None)

#mixed_tensor, label = augment_and_mix(background, drone, noise, sr, save_path="augmented_mixed.wav")

gorillaz, sr = librosa.load("Gorillaz - Clint Eastwood.wav", sr=16000)
os.makedirs("trial", exist_ok=True)

for i in range(10):
    augmented_audio, log = strong_augmentation(gorillaz, sr)
    out_wav = os.path.join("trial", f"augmented_gorillaz_{i+1}.wav")
    out_txt = os.path.join("trial", f"augmented_gorillaz_{i+1}_log.txt")
    
    sf.write(out_wav, augmented_audio, sr)
    
    with open(out_txt, 'w') as f:
        f.write("Augmentation steps applied (in order):\n")
        for step in log:
            f.write(step + "\n")
    
    print(f"Saved audio: {out_wav}")
    print(f"Saved log: {out_txt}")
