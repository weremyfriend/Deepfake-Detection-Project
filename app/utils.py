import os
from PIL import Image, UnidentifiedImageError


def clean_dataset(folder_path):
    """
    Clean up dataset by removing corrupted or invalid files
    
    This function walks through a folder and:
    1. Removes non-image files
    2. Removes corrupted images that can't be opened
    3. Keeps .gitkeep files for version control
    """
    # List of valid image extensions

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif') # Change when adding other media types for video support
    
    # Walk through all subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip .gitkeep files! Keep them in the repo so it's eaiser
            if file == ".gitkeep":
                continue
            
            # Check file extension
            if not file.lower().endswith(valid_extensions):
                print(f"Removing non-image: {file_path}")
                os.remove(file_path)
                continue
            
            # Try to open and verify the image
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Checks if image is corrupted
            except (UnidentifiedImageError, OSError):
                # Image is corrupt
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)