import os
import shutil

import os
import shutil
import hashlib
import random

def file_hash(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        buf = f.read(65536)
        while buf:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def unpack_folders_skip_duplicates(src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    existing_hashes = {}
    
    # Precompute hashes for files already in destination
    for f in os.listdir(dest_folder):
        dest_file_path = os.path.join(dest_folder, f)
        if os.path.isfile(dest_file_path):
            existing_hashes[file_hash(dest_file_path)] = dest_file_path
    
    files_found = 0
    files_copied = 0
    
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            files_found += 1
            src_file_path = os.path.join(root, file)
            src_hash = file_hash(src_file_path)
            
            if src_hash in existing_hashes:
                print(f"Skipping duplicate file (content exists): {src_file_path}")
                continue  # skip duplicate content
            
            parent_folder = os.path.basename(os.path.dirname(src_file_path))
            new_file_name = f"{parent_folder}_{file}"
            dest_file_path = os.path.join(dest_folder, new_file_name)
            
            count = 1
            base, ext = os.path.splitext(new_file_name)
            while os.path.exists(dest_file_path):
                dest_file_path = os.path.join(dest_folder, f"{base}_{count}{ext}")
                count += 1
            
            shutil.copy2(src_file_path, dest_file_path)
            files_copied += 1
            existing_hashes[src_hash] = dest_file_path
            print(f"Copied: {src_file_path} -> {dest_file_path}")
    
    if files_found == 0:
        print("No files found in the source folder.")
    else:
        print(f"Finished copying {files_copied} files from {files_found} found.")


def split_and_copy_dataset(src_folder, dest_training, dest_validation, dest_test, train_ratio=0.7, val_ratio=0.3, test_ratio=0.0):
    os.makedirs(dest_training, exist_ok=True)
    os.makedirs(dest_validation, exist_ok=True)
    os.makedirs(dest_test, exist_ok=True)

    all_files = []
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            all_files.append(os.path.join(root, file))

    random.shuffle(all_files)
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    print(f"Total files: {n_total}, Training: {n_train}, Validation: {n_val}, Test: {n_test}")
    if n_total == 0:
        print("No files found in the source folder.")
        return
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]

    for f in train_files:
        shutil.copy2(f, os.path.join(dest_training, os.path.basename(f)))
    for f in val_files:
        shutil.copy2(f, os.path.join(dest_validation, os.path.basename(f)))
    for f in test_files:
        shutil.copy2(f, os.path.join(dest_test, os.path.basename(f)))

    print(f"Copied {len(train_files)} files to training, {len(val_files)} to validation, {len(test_files)} to test.")

import subprocess
import json

#uch sie promptowania kurde
#For each file listed in the file_path, run: ffprobe -show_format -show_streams $FILENAME and report on these values: bitrate, duration, bits_per_sample (bits per sample), sample rate, channels (mono vs stereo). Aggregate data from every file and display ranges present in the dataset such that I can compare and investigate outliers
#oczyswiscie ze nie dzialalo wiec promptuje dalej
#traversovanie przez pelno plikow uzywajac komend basha po drodze jest dosyc czasochlonny, ale zawsze dziala pieknie
"""wyniki dla Background:
=== Aggregated Dataset Statistics ===
Bitrate: 256035 - 256352 bps
Duration: 1.0 - 10.0 s
Bit Depth(bits per sample): 16 - 16
Sample Rate: 16000 - 16000 Hz
Channels: {'mono'}

wyniki dla dataset_preliminary:
=== Aggregated Dataset Statistics ===
Bitrate: 256609 - 257329 bps
Duration: 0.649938 - 1.024 s
Bit Depth(bits per sample): 16 - 16
Sample Rate: 16000 - 16000 Hz
Channels: {'mono'}

wyniki dla noise/
=== Aggregated Dataset Statistics ===
Bitrate: 256121 - 672000 bps
Duration: 0.0015 - 5.12 s
Bit Depth(bits per sample): 16 - 16
Sample Rate: 16000 - 16000 Hz
Channels: {'mono'}


ffprobe.json to przyklkadowy output jaki dostalem z funckji subprocess.run(cmd)
"""
def analyze_audio_files(file_paths):
    stats = {
        "bitrate": [],
        "duration": [],
        "bits_per_sample": [],
        "sample_rate": [],
        "channels": [],
    }
    channel_map = {1: "mono", 2: "stereo"}

    # If file_paths is a folder name, collect all files recursively
    if isinstance(file_paths, str) and os.path.isdir(file_paths):
        collected_files = []
        for root, dirs, files in os.walk(file_paths):
            for file in files:
                collected_files.append(os.path.join(root, file))
        file_paths = collected_files

    for file_path in file_paths:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-of", "json",
            file_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            # Save the JSON output for inspection if it's the first time for this file
            path = os.path.dirname(file_path)
            json_save_path = "ffprobe.json"
            if not os.path.exists(json_save_path):
                with open(json_save_path, "w") as jf:
                    jf.write(result.stdout)
            format_info = info.get("format", {})
            streams = info.get("streams", [])
            stream_info = streams[0] if streams else {}

            bitrate = int(format_info.get("bit_rate", 0))
            duration = float(format_info.get("duration", 0))
            bits_per_sample = int(stream_info.get("bits_per_sample", 0)) if "bits_per_sample" in stream_info else None
            sample_rate = int(stream_info.get("sample_rate", 0))
            channels = int(stream_info.get("channels", 0))

            stats["bitrate"].append(bitrate)
            stats["duration"].append(duration)
            stats["bits_per_sample"].append(bits_per_sample)
            stats["sample_rate"].append(sample_rate)
            stats["channels"].append(channels)

            print(f"File: {file_path}")
            print(f"  Bitrate: {bitrate} bps")
            print(f"  Duration: {duration:.2f} s")
            print(f"  Bit Depth: {bits_per_sample if bits_per_sample else 'N/A'}")
            print(f"  Sample Rate: {sample_rate} Hz")
            print(f"  Channels: {channels} ({channel_map.get(channels, 'other')})")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            print("Stopping execution due to error.")
            print(f"Problematic file: {file_path}")
            return

    def range_or_set(values):
        vals = [v for v in values if v]
        if not vals:
            return "N/A"
        if all(isinstance(v, int) or isinstance(v, float) for v in vals):
            return f"{min(vals)} - {max(vals)}"
        return set(vals)

    print("\n=== Aggregated Dataset Statistics ===")
    print(f"Bitrate: {range_or_set(stats['bitrate'])} bps")
    print(f"Duration: {range_or_set(stats['duration'])} s")
    print(f"Bit Depth(bits per sample): {range_or_set(stats['bits_per_sample'])}")
    print(f"Sample Rate: {range_or_set(stats['sample_rate'])} Hz")
    channel_types = set(channel_map.get(c, f"{c}ch") for c in stats["channels"] if c)
    print(f"Channels: {channel_types if channel_types else 'N/A'}")

def delete_json_files(folder):
    count = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    count += 1
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    print(f"Deleted {count} .json files from {folder}")

if __name__ == "__main__":
    # src_folder = "MAD_dataset/training"
    # dest_folder = "Background"
    #unpack_folders_skip_duplicates(src_folder, dest_folder)

    # training_folder = "dataset_current/training"
    # validation_folder = "dataset_current/validation"
    # test_folder = "dataset_current/test"
    # src_folder = "dataset_preliminary"
    # split_and_copy_dataset(src_folder, training_folder, validation_folder, test_folder)

    analyze_audio_files("noise")
    # delete_json_files("dataset_preliminary")