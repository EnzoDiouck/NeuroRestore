import os
import urllib.request
import zipfile
from pathlib import Path
import glob
from PIL import Image
import matplotlib.pyplot as plt

def fetch_dataset(dataset_url: str, dataset_file: str, output_dir: str):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if file already exists
    if os.path.exists(dataset_file):
        file_size = os.path.getsize(dataset_file)
        print(f"✓ Dataset file already exists: {dataset_file}")
        print(f"✓ File size: {file_size / (1024*1024):.2f} MB")
        print("Skipping download...")
    else:
        # Download the dataset
        print(f"Downloading dataset from {dataset_url}...")
        print(f"This may take a while depending on your connection speed...")

        try:
            # Download with progress
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100.0 / total_size, 100)
                print(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end='')

            urllib.request.urlretrieve(dataset_url, dataset_file, reporthook=download_progress)
            print("\n✓ Download complete!")

        except Exception as e:
            print(f"\n✗ Error downloading dataset: {e}")
            raise

    # Verify the downloaded file
    if os.path.exists(dataset_file):
        file_size = os.path.getsize(dataset_file)
        print(f"✓ File exists: {dataset_file}")
        print(f"✓ File size: {file_size / (1024*1024):.2f} MB")
    else:
        print(f"✗ File not found: {dataset_file}")
        raise FileNotFoundError(f"Dataset file {dataset_file} was not downloaded successfully")



    # Extract the zip file
    print(f"Extracting {dataset_file} to {output_dir}...")

    try:
        if len(os.listdir(output_dir)) == 0:
            with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
                # Get total number of files
                total_files = len(zip_ref.namelist())
                print(f"Extracting {total_files} files...")

                # Extract all files
                zip_ref.extractall(output_dir)

            print("✓ Extraction complete!")
        else:
            print("✓ Extraction already done!")


        # Optional: Remove zip file to save space
        # os.remove(DATASET_FILE)
        # print(f"✓ Removed {DATASET_FILE}")

    except Exception as e:
        print(f"✗ Error extracting dataset: {e}")
        raise




    # Explore the extracted dataset structure
    print(f"Dataset directory structure:\n")

    # List top-level directories
    for item in sorted(os.listdir(output_dir)):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            # Count files in subdirectories
            file_count = len(glob.glob(os.path.join(item_path, '**', '*.*'), recursive=True))
            print(f"📁 {item}/ ({file_count} files)")
        else:
            file_size = os.path.getsize(item_path) / (1024*1024)
            print(f"📄 {item} ({file_size:.2f} MB)")

    # Look for image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']
    total_images = 0

    for ext in image_extensions:
        images = glob.glob(os.path.join(output_dir, '**', ext), recursive=True)
        if images:
            print(f"\n✓ Found {len(images)} {ext.upper()} images")
            total_images += len(images)

    print(f"\nTotal images found: {total_images}")




def display_sample_images(output_dir: str, num_samples: int = 6):
    # Find all images
    all_images = glob.glob(os.path.join(output_dir, '**', '*.jpg'), recursive=True)

    if len(all_images) > 0:
        # Display first 6 images
        num_samples = min(6, len(all_images))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(num_samples):
            img = Image.open(all_images[i])
            axes[i].imshow(img)
            axes[i].set_title(f"Image {i+1}: {os.path.basename(all_images[i])}")
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(num_samples, 6):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"\nDisplayed {num_samples} sample images from the dataset")
    else:
        print("No images found in the dataset")