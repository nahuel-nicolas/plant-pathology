import os
import shutil
import pandas as pd

# Check if data/train/ already exists
if not os.path.exists('data/train/'):
    print("data/train/ doesn't exist. Creating folder structure and copying images...")

    # Read the train.csv file
    train_df = pd.read_csv('plant-pathology-2020-fgvc7/train.csv')

    # Label columns
    label_columns = ['healthy', 'multiple_diseases', 'rust', 'scab']

    # Create base directory
    os.makedirs('data/train/', exist_ok=True)

    # Create label subdirectories
    for label in label_columns:
        os.makedirs(f'data/train/{label}/', exist_ok=True)

    # Process each image
    for idx, row in train_df.iterrows():
        image_id = row['image_id']

        # Find which label is 1 (active)
        for label in label_columns:
            if row[label] == 1:
                # Source and destination paths
                src_path = f'plant-pathology-2020-fgvc7/images/{image_id}.jpg'
                dst_path = f'data/train/{label}/{image_id}.jpg'

                # Copy the image
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    if (idx + 1) % 100 == 0:
                        print(f"Processed {idx + 1} images...")
                else:
                    print(f"Warning: {src_path} not found")

                break  # Only one label should be active per image

    print(f"Done! Organized {len(train_df)} images into label folders.")

    # Print summary
    print("\nSummary:")
    for label in label_columns:
        count = len(os.listdir(f'data/train/{label}/'))
        print(f"  {label}: {count} images")
else:
    print("data/train/ already exists. Skipping image organization.")
