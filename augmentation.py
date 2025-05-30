import numpy as np
import librosa
import soundfile as sf
from scipy.stats import truncnorm
import os

def truncated_normal(mean, std, lower, upper):
    return truncnorm(
        (lower - mean) / std, (upper - mean) / std, loc=mean, scale=std).rvs()

def amplitude_clipping(audio, clip_range=(0.75, 1.0)):
    clip_amp = truncated_normal(0.875, 0.05, *clip_range)
    max_val = np.max(np.abs(audio))
    clipped = np.clip(audio, -clip_amp * max_val, clip_amp * max_val)
    return clipped

def volume_amplify(audio, amp_range=(0.5, 1.5)):
    gain = truncated_normal(1.0, 0.25, *amp_range)
    return audio * gain

def add_echo(audio, sr, delay_range=(0.02, 0.4), decay=0.5):
    delay_sec = truncated_normal(0.21, 0.1, *delay_range)
    delay_samples = int(delay_sec * sr)
    echo = np.zeros_like(audio)
    echo[delay_samples:] = audio[:-delay_samples] * decay
    return audio + echo

#def pitch_shift(audio, sr, pitch_range=(0, 4)):
 #   n_steps = int(truncated_normal(2, 1.5, *pitch_range))
  #  return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)

def partial_erase(audio, erase_ratio=(0, 0.3)):
    erase_fraction = truncated_normal(0.15, 0.1, *erase_ratio)
    erase_len = int(len(audio) * erase_fraction)
    start = np.random.randint(0, len(audio) - erase_len)
    audio[start:start + erase_len] = np.random.normal(0, 0.01, erase_len)
    return audio

#def speed_adjust(audio, speed_range=(0.5, 1.5)):
 #   speed = truncated_normal(1.0, 0.25, *speed_range)
  #  return librosa.effects.time_stretch((audio, speed))

def add_noise(audio, noise_level=(0.001, 0.02)):
    noise_amp = truncated_normal(0.01, 0.005, *noise_level)
    return audio + np.random.normal(0, noise_amp, size=len(audio))

def hpss_separate(audio):
    harm, perc = librosa.effects.hpss(audio)
    return harm + perc

def augment_audio(file_path, output_prefix, num_versions=5):
    y, sr = librosa.load(file_path, sr=None)
    for i in range(num_versions):
        audio = y.copy()
        audio = amplitude_clipping(audio)
        audio = volume_amplify(audio)
        audio = add_echo(audio, sr)
        #audio = pitch_shift(audio, sr)
        audio = partial_erase(audio)
        #audio = speed_adjust(audio)
        audio = add_noise(audio)
        audio = hpss_separate(audio)

        output_file = f"{output_prefix}_aug{i+1}.wav"
        sf.write(output_file, audio, sr)
        print(f"Saved: {output_file}")

## Example usage
### augment_audio("B_S2_D1_067-bebop_000_.wav", "Augmented")
