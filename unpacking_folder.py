import os
import shutil

import os
import shutil
import hashlib

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

# Usage example:
src_folder = "MAD_dataset/training"
dest_folder = "Background"

unpack_folders_skip_duplicates(src_folder, dest_folder)