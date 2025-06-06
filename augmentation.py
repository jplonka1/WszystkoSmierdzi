import numpy as np
import librosa
import torch
import soundfile as sf
import os
import sys
import random
from scipy.signal import butter, lfilter
from scipy.stats import truncnorm
from typing import Optional
import uuid
import json

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

### AUGMENTACJA

# -------- Utility for truncated normal --------
def truncated_normal(mean, std, lower, upper):
    return truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std).rvs()

# -------- Augmentation Functions --------
def amplitude_clipping(audio, percent):
    # percent: 0-100, how much of the amplitude to keep (e.g. 15 means clip at 85% of max)
    clip_amp = 1.0 - percent / 100.0
    max_val = np.max(np.abs(audio))
    return np.clip(audio, -clip_amp * max_val, clip_amp * max_val)

def volume_amplify(audio, gain_db):
    gain_lin = 10 ** (gain_db / 20)
    return audio * gain_lin

def add_echo(audio, sr, delay_ms, decay):
    delay_samples = int((delay_ms / 1000.0) * sr)
    echo = np.zeros_like(audio)
    if delay_samples < len(audio):
        echo[delay_samples:] = audio[:-delay_samples] * decay
    return audio + echo

def partial_erase(audio, sr, erase_fraction):
    audio = audio.copy()
    total_len = len(audio)
    total_erase_samples = int(erase_fraction * total_len)
    erased = 0
    attempts = 0
    max_attempts = 1000
    while erased < total_erase_samples and attempts < max_attempts:
        chunk_len = np.random.randint(int(0.01 * total_len), int(0.05 * total_len))
        if chunk_len == 0 or chunk_len + erased > total_erase_samples or chunk_len >= total_len:
            attempts += 1
            continue
        start = np.random.randint(0, total_len - chunk_len)
        audio[start:start + chunk_len] = np.random.normal(0, 0.01, chunk_len)
        erased += chunk_len
        attempts += 1
    return audio

def pitch_shift(audio, sr, semitones):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)

def add_noise(audio, snr_db):
    # SNR in dB: higher means less noise
    rms_signal = np.sqrt(np.mean(audio ** 2))
    snr_linear = 10 ** (snr_db / 20)
    noise_rms = rms_signal / snr_linear
    noise = np.random.normal(0, noise_rms, size=len(audio))
    return audio + noise

def hpss_separate(audio, harmonic_scale, percussive_scale):
    harm, perc = librosa.effects.hpss(audio)
    augmented = harmonic_scale * harm + percussive_scale * perc
    max_val = np.max(np.abs(augmented))
    if max_val > 1.0:
        augmented = augmented / max_val
    return augmented

def time_shift(audio, sr, max_shift_seconds):
    max_shift = int(max_shift_seconds * sr)
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(audio, shift)

def tanh_distortion(audio, strength):
    return np.tanh(audio * (1.0 + strength * 10))

def butter_filter(audio, sr, cutoff, btype='low', order=5):
    nyquist = 0.5 * sr
    norm_cutoff = np.array(cutoff) / nyquist
    b, a = butter(order, norm_cutoff, btype=btype, analog=False)
    return lfilter(b, a, audio)

def high_pass(audio, sr, cutoff):
    return butter_filter(audio, sr, cutoff, btype='high')

def band_pass(audio, sr, low, high):
    return butter_filter(audio, sr, [low, high], btype='band')

def random_filter(audio, sr, config):
    # config: dict with keys 'p_apply', 'high_pass', 'band_pass'
    if random.random() > config.get('p_apply', 0.8):
        return audio
    choice = random.choice(['high', 'band'])
    if choice == 'high':
        hp = config['high_pass']
        cutoff = truncated_normal(hp['mean'], hp['std'], *hp['range'])
        return high_pass(audio, sr, cutoff)
    else:
        bp = config['band_pass']
        low = truncated_normal(bp['low_mean'], bp['low_std'], *bp['low_range'])
        bw = truncated_normal(bp['bw_mean'], bp['bw_std'], *bp['bw_range'])
        high = min(sr // 2 - 1, low + bw)
        return band_pass(audio, sr, low, high)

# -------- Augmentation Wrappers --------
def get_augmentations_for_type(audio_type, config):
    mild = config['augmentations']['mild']
    strong = config['augmentations']['strong']
    mild_augmentations = [
        ('amplitude_clipping', amplitude_clipping, mild.get('amplitude_clipping')),
        ('volume_amplify', volume_amplify, mild.get('volume_amplify')),
        ('add_echo', add_echo, mild.get('add_echo')),
        ('partial_erase', partial_erase, mild.get('partial_erase')),
        ('add_noise', add_noise, mild.get('add_noise')),
    ]
    strong_augmentations = [
        ('amplitude_clipping', amplitude_clipping, strong.get('amplitude_clipping')),
        ('volume_amplify', volume_amplify, strong.get('volume_amplify')),
        ('add_echo', add_echo, strong.get('add_echo')),
        ('partial_erase', partial_erase, strong.get('partial_erase')),
        ('add_noise', add_noise, strong.get('add_noise')),
        ('pitch_shift', pitch_shift, strong.get('pitch_shift')),
        ('hpss_separate', hpss_separate, strong.get('hpss_separate')),
        ('time_shift', time_shift, strong.get('time_shift')),
        ('tanh_distortion', tanh_distortion, strong.get('tanh_distortion')),
        ('random_filter', random_filter, strong.get('random_filter')),
    ]
    if audio_type == "background":
        return mild_augmentations
    elif audio_type == "sound":
        return strong_augmentations
    else:
        return strong_augmentations

def apply_augmentation(audio, sr, audio_type, config):
    augmentations = get_augmentations_for_type(audio_type, config)
    random.shuffle(augmentations)
    log = []
    audio_out = audio.copy()
    for name, func, aug_cfg in augmentations:
        if aug_cfg is None:
            continue
        prob = aug_cfg.get('probability', 1.0)
        if random.random() > prob:
            continue
        if name == 'amplitude_clipping':
            p = aug_cfg['clip_percent']
            percent = truncated_normal(p['mean'], p['std'], *p['range'])
            audio_out = amplitude_clipping(audio_out, percent)
            log.append(f"{name}(clip_percent={percent:.2f})")
        elif name == 'volume_amplify':
            p = aug_cfg['gain_db']
            gain_db = truncated_normal(p['mean'], p['std'], *p['range'])
            audio_out = volume_amplify(audio_out, gain_db)
            log.append(f"{name}(gain_db={gain_db:.2f})")
        elif name == 'add_echo':
            delay_cfg = aug_cfg['delay_ms']
            decay_cfg = aug_cfg['decay']
            delay_ms = truncated_normal(delay_cfg['mean'], delay_cfg['std'], *delay_cfg['range'])
            decay = truncated_normal(decay_cfg['mean'], decay_cfg['std'], *decay_cfg['range'])
            audio_out = add_echo(audio_out, sr, delay_ms, decay)
            log.append(f"{name}(delay_ms={delay_ms:.2f}, decay={decay:.2f})")
        elif name == 'partial_erase':
            p = aug_cfg['erase_fraction']
            erase_fraction = truncated_normal(p['mean'], p['std'], *p['range'])
            audio_out = partial_erase(audio_out, sr, erase_fraction)
            log.append(f"{name}(erase_fraction={erase_fraction:.3f})")
        elif name == 'add_noise':
            p = aug_cfg['snr_db']
            snr_db = truncated_normal(p['mean'], p['std'], *p['range'])
            audio_out = add_noise(audio_out, snr_db)
            log.append(f"{name}(snr_db={snr_db:.2f})")
        elif name == 'pitch_shift':
            if aug_cfg is None:
                continue
            p = aug_cfg['semitones']
            semitones = truncated_normal(p['mean'], p['std'], *p['range'])
            audio_out = pitch_shift(audio_out, sr, semitones)
            log.append(f"{name}(semitones={semitones:.2f})")
        elif name == 'hpss_separate':
            harm_cfg = aug_cfg['harmonic_scale']
            perc_cfg = aug_cfg['percussive_scale']
            harmonic_scale = truncated_normal(harm_cfg['mean'], harm_cfg['std'], *harm_cfg['range'])
            percussive_scale = truncated_normal(perc_cfg['mean'], perc_cfg['std'], *perc_cfg['range'])
            audio_out = hpss_separate(audio_out, harmonic_scale, percussive_scale)
            log.append(f"{name}(harmonic_scale={harmonic_scale:.2f}, percussive_scale={percussive_scale:.2f})")
        elif name == 'time_shift':
            p = aug_cfg['max_shift_seconds']
            max_shift_seconds = truncated_normal(p['mean'], p['std'], *p['range'])
            audio_out = time_shift(audio_out, sr, max_shift_seconds)
            log.append(f"{name}(max_shift_seconds={max_shift_seconds:.3f})")
        elif name == 'tanh_distortion':
            p = aug_cfg['strength']
            strength = truncated_normal(p['mean'], p['std'], *p['range'])
            audio_out = tanh_distortion(audio_out, strength)
            log.append(f"{name}(strength={strength:.2f})")
        elif name == 'random_filter':
            audio_out = random_filter(audio_out, sr, aug_cfg)
            log.append(f"{name}(params=custom)")
    return audio_out, log

# -------- Main Augmentation Logic --------
def augment_and_mix(background, sounds, sr, config, save_path=None):
    """
    background: np.ndarray, main background audio
    sounds: list of np.ndarray, each sound to be mixed in (augmented identically)
    sr: sample rate
    config: dict, configuration for augmentations
    save_path: optional, if provided saves the result and logs
    Returns: mixed_tensor (torch.Tensor)
    """
    # Augment background (mild)
    background_aug, background_log = apply_augmentation(background.copy(), sr, "background", config)

    # If there are no sounds, just return background
    if not sounds or len(sounds) == 0:
        mixed_tensor = torch.from_numpy(background_aug).float()
        if save_path:
            filename = os.path.splitext(os.path.basename(save_path))[0]
            out_wav = os.path.join(os.path.dirname(save_path), f"{filename}.wav")
            os.makedirs(os.path.dirname(out_wav), exist_ok=True)
            sf.write(out_wav, background_aug, sr)
        return mixed_tensor

    # Augment all sounds identically: generate one augmentation log, apply to all
    ref_sound = sounds[0].copy()
    aug_ref, aug_log = apply_augmentation(ref_sound, sr, "sound", config)
    sounds_aug = [aug_ref.copy() for _ in sounds]

    # Mix all sounds into background at random positions independently
    mixed = background_aug.copy()
    for aug_sound in sounds_aug:
        mixed = insert_with_fade(mixed, aug_sound, sr, fade=True, gain_db_range=(-4, 0))

    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1:
        mixed /= peak

    mixed_tensor = torch.from_numpy(mixed).float()

    # If save_path is provided, save the mixed audio and logs
    if save_path:
        filename = os.path.splitext(os.path.basename(save_path))[0]
        out_wav = os.path.join(os.path.dirname(save_path), f"{filename}.wav")
        os.makedirs(os.path.dirname(out_wav), exist_ok=True)
        sf.write(out_wav, mixed, sr)
        print(f"[DEBUG] Saved: {out_wav}")

        # Save logs for each component (sound log only once)
        log_folder = os.path.dirname(save_path)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder, exist_ok=True)
        log_path = os.path.join(log_folder, f"{filename}_log.txt")
        with open(log_path, "w") as f:
            f.write("Augmentation steps applied (in order):\n")
            f.write("Background:\n")
            for step in background_log:
                f.write("  " + step + "\n")
            f.write("Sound (applied identically to all):\n")
            for step in aug_log:
                f.write("  " + step + "\n")
        print(f"[DEBUG] Saved log: {log_path}")

    return mixed_tensor

## -------- Example Usage --------
# Example usage of augment_and_mix function

if __name__ == "__main__":
    # Load config from JSON file once
    with open("config_default.json", "r") as f:
        config = json.load(f)

    for i in range(10):
        # Load audio files from disk before passing to augment_and_mix
        background_path = "001_0.wav"
        sound_paths = ["1-11687-A-472.wav", "1-137-A-320.wav", "B_S2_D1_067-bebop_001_.wav"]
        sr = 16000

        background, _ = librosa.load(background_path, sr=sr)
        sounds = [librosa.load(p, sr=sr)[0] for p in sound_paths]

        # Generate a random unique filename for each run
        random_id = uuid.uuid4().hex[:8]
        save_path = f"output/augmented_mixed_{random_id}_{i+1}.wav"

        augment_and_mix(background, sounds, sr, config, save_path=save_path)

### TESTING AUGMENTATION FUNCTIONS WITH GORILLAZ AUDIO
        #gorillaz_path = "Gorillaz - Clint Eastwood.wav"
        #gorillaz, sr = librosa.load(gorillaz_path, sr=16000)
        #os.makedirs("trial", exist_ok=True)

        # Test mild augmentations
        #for i in range(3):
        #    augmented_audio, log = apply_augmentation(gorillaz, sr, "background", config)
        #    out_wav = os.path.join("trial", f"gorillaz_mild_{i+1}.wav")
        #    out_txt = os.path.join("trial", f"gorillaz_mild_{i+1}_log.txt")
        #    sf.write(out_wav, augmented_audio, sr)
        #    with open(out_txt, 'w') as f:
        #        f.write("Mild augmentation steps applied (in order):\n")
        #        for step in log:
        #            f.write(step + "\n")
        #    print(f"Saved mild audio: {out_wav}")
        #    print(f"Saved mild log: {out_txt}")

        # Test strong augmentations
        #for i in range(3):
        #    augmented_audio, log = apply_augmentation(gorillaz, sr, "sound", config)
        #    out_wav = os.path.join("trial", f"gorillaz_strong_{i+1}.wav")
        #    out_txt = os.path.join("trial", f"gorillaz_strong_{i+1}_log.txt")
        #    sf.write(out_wav, augmented_audio, sr)
        #    with open(out_txt, 'w') as f:
        #        f.write("Strong augmentation steps applied (in order):\n")
        #        for step in log:
        #            f.write(step + "\n")
        #    print(f"Saved strong audio: {out_wav}")
        #    print(f"Saved strong log: {out_txt}")
