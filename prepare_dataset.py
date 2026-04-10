import os
import shutil
import kagglehub
import yaml

def prepare_yolo_dataset():
    print("Downloading dataset...")
    # Download dataset
    path = kagglehub.dataset_download("anushkagovindkadam/smart-helmet-detection-using-dl")
    
    # Original paths inside the kaggle repository
    orig_train_images = os.path.join(path, "data", "train", "images")
    orig_train_labels = os.path.join(path, "data", "train", "labels")
    orig_val_images = os.path.join(path, "data", "vaid", "images")
    orig_val_labels = os.path.join(path, "data", "vaid", "labels")
    
    # Target paths in your local project
    target_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset"))
    
    # Clean target base if exists
    if os.path.exists(target_base):
        shutil.rmtree(target_base)
        
    for split in ["train", "val"]:
        os.makedirs(os.path.join(target_base, "images", split), exist_ok=True)
        os.makedirs(os.path.join(target_base, "labels", split), exist_ok=True)
        
    # Map original classes to 3 unified classes:
    # Original:
    # 0: driver_with_helmet, 1: bike, 2: driver, 3: passenger_with_helemt, 
    # 4: passenger, 5: driver_without_helmet, 6: passenger_without_helemt
    class_mapping = {
        0: 0, # helmet
        1: None, # drop bike to focus on persons and helmets
        2: 2, # rider
        3: 0, # helmet
        4: 2, # rider
        5: 1, # no_helmet
        6: 1, # no_helmet
    }

    def process_split(orig_img_dir, orig_lbl_dir, target_split):
        target_img_dir = os.path.join(target_base, "images", target_split)
        target_lbl_dir = os.path.join(target_base, "labels", target_split)
        
        for img_name in os.listdir(orig_img_dir):
            if not img_name.endswith(('.jpg', '.jpeg', '.png')): 
                continue
                
            shutil.copy2(os.path.join(orig_img_dir, img_name), os.path.join(target_img_dir, img_name))
            
            # Process label mapping
            base_name = os.path.splitext(img_name)[0]
            lbl_name = base_name + ".txt"
            
            orig_lbl_path = os.path.join(orig_lbl_dir, lbl_name)
            target_lbl_path = os.path.join(target_lbl_dir, lbl_name)
            
            if os.path.exists(orig_lbl_path):
                with open(orig_lbl_path, "r") as f:
                    lines = f.readlines()
                    
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    class_id = int(parts[0])
                    new_id = class_mapping.get(class_id, None)
                    if new_id is not None:
                        new_lines.append(f"{new_id} " + " ".join(parts[1:]) + "\n")
                        
                with open(target_lbl_path, "w") as f:
                    f.writelines(new_lines)
            else:
                # create empty label file for background images
                open(target_lbl_path, "w").close()

    print("Processing training split...")
    process_split(orig_train_images, orig_train_labels, "train")
    print("Processing validation split...")
    process_split(orig_val_images, orig_val_labels, "val")
    
    # Write data.yaml for YOLO training
    data_yaml = {
        "path": target_base,
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": ["helmet", "no_helmet", "rider"]
    }
    
    with open(os.path.join(target_base, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    print(f"Dataset prepared successfully at: {target_base}")
    print("Classes mapped to:")
    for idx, name in enumerate(data_yaml["names"]):
        print(f"  {idx}: {name}")

if __name__ == "__main__":
    prepare_yolo_dataset()
