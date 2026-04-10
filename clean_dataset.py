import os
from pathlib import Path

# Dataset dir
DATASET_DIR = Path('/Users/tania/Desktop/eagle_eye/dataset')

# Subdirectories to check
splits = ['train', 'val']

removed_files = 0
total_files = 0

def check_and_delete(label_path, image_extensions=['.jpg', '.jpeg', '.png']):
    global removed_files
    
    # Try to find corresponding image file
    image_dir = label_path.parent.parent.parent / 'images' / label_path.parent.name
    image_base = label_path.stem
    image_path = None
    
    for ext in image_extensions:
        img = image_dir / (image_base + ext)
        if img.exists():
            image_path = img
            break
            
    is_invalid = False
    
    if not label_path.exists() or label_path.stat().st_size == 0:
        is_invalid = True
    else:
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            if len(lines) == 0:
                is_invalid = True
            else:
                for line in lines:
                    parts = line.strip().split()
                    
                    if len(parts) < 5:
                        is_invalid = True
                        break
                        
                    class_id = int(parts[0])
                    if class_id not in [0, 1]:
                        is_invalid = True
                        break
                        
                    # coordinates
                    coords = [float(x) for x in parts[1:5]]
                    for c in coords:
                        if c < 0 or c > 1:
                            is_invalid = True
                            break
                    
                    if is_invalid:
                        break
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            is_invalid = True
            
    if is_invalid:
        print(f"Deleting invalid label: {label_path}")
        if label_path.exists():
            label_path.unlink()
        if image_path and image_path.exists():
            print(f"Deleting corresponding image: {image_path}")
            image_path.unlink()
        removed_files += 1

def main():
    global total_files
    
    for split in splits:
        labels_dir = DATASET_DIR / 'labels' / split
        
        if not labels_dir.exists():
            continue
            
        for label_file in labels_dir.glob('*.txt'):
            total_files += 1
            check_and_delete(label_file)
            
    print("-" * 30)
    print(f"Number of removed files: {removed_files}")
    print(f"Final dataset size (labels remaining): {total_files - removed_files}")

if __name__ == '__main__':
    main()
