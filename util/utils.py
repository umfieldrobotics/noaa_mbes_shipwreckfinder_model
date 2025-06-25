import os
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont

# Clear directories to regenerate file distribution
def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def save_combined_image(image, pred, label, test_file, pred_path, ship_iou=None, terrain_iou=None, divider_width=5):    
    """
    Saves a single image containing the original image, prediction, and label side by side,
    with thin white vertical dividers between them, and overlays IoU scores.

    Parameters:
    - image: PyTorch tensor (C, H, W) - The input image.
    - pred: NumPy array (H, W) - The predicted mask.
    - label: NumPy array (H, W) - The ground truth mask.
    - test_file: str - The original file path (used for naming output file).
    - pred_path: str - Directory to save the output image.
    - ship_iou: float - Ship IoU score to overlay.
    - terrain_iou: float - Terrain IoU score to overlay.
    - divider_width: int - Width of the divider lines.
    """

    # Convert the image (C, H, W) tensor to (H, W, C) NumPy array and scale to [0,255]
    img = image.squeeze().cpu().numpy()  # Convert to (H, W, C)
    img = (255 * img).astype(np.uint8)  # Scale to [0, 255]
    img_pil = Image.fromarray(img)  # Convert to PIL image

    pred_img = Image.fromarray((255 * pred[0, ...]).astype(np.uint8))
    label_img = Image.fromarray((255 * label.squeeze()).astype(np.uint8))
    # pred_img = Image.fromarray((127.5 * pred[0, ...]+127.5).astype(np.uint8))
    # label_img = Image.fromarray((127.5*label.squeeze()+127.5).astype(np.uint8))

    # Ensure all images have the same height
    height = img_pil.height
    pred_img = pred_img.resize((pred_img.width, height))
    label_img = label_img.resize((label_img.width, height))

    # Create a white divider
    divider = Image.new("RGB", (divider_width, height), color=(128, 128, 128))

    # Calculate final image width: 3 images + 2 dividers
    combined_width = img_pil.width + pred_img.width + label_img.width + (2 * divider_width)
    combined_img = Image.new("RGB", (combined_width, height), color=(0, 0, 0))

    # Paste images with dividers in between
    x_offset = 0
    combined_img.paste(img_pil, (x_offset, 0))
    x_offset += img_pil.width
    combined_img.paste(divider, (x_offset, 0))
    x_offset += divider_width
    combined_img.paste(label_img.convert("RGB"), (x_offset, 0))
    x_offset += pred_img.width
    combined_img.paste(divider, (x_offset, 0))
    x_offset += divider_width
    combined_img.paste(pred_img.convert("RGB"), (x_offset, 0))

    # Draw IoU text at bottom center
    if ship_iou is not None or terrain_iou is not None:
        draw = ImageDraw.Draw(combined_img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        text = f"Ship IoU: {ship_iou:.4f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (combined_img.width - text_width) // 2
        text_y = combined_img.height - text_height - 10  # Slight margin from bottom

        # Add black rectangle for better visibility
        padding = 5
        draw.rectangle(
            [text_x - padding, text_y - padding, text_x + text_width + padding, text_y + text_height + padding],
            fill=(255, 255, 255)
        )

        draw.text((text_x, text_y), text, font=font, fill=(255, 0, 0))  # Red text

    # Save the final combined image
    output_filename = os.path.basename(test_file).replace("_image.npy", "_combined.png")
    combined_img.save(os.path.join(pred_path, output_filename))
    
    return combined_img

def save_plot(save_path, data):
    # Create the figuredepth_grid
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray', interpolation='nearest')
    plt.colorbar()  # Optional: add a colorbar
    plt.axis('off')  # Optional: remove axes for a cleaner look

    # Save the figure
    plt.savefig(save_path)

    # Show the figure (optional)
    # plt.show()

def normalize_nonzero(image):
    """
    Normalizes nonzero values in an image (tensor or numpy array) to the range [0,1].
    
    Parameters:
        image (torch.Tensor or np.ndarray): Input image.

    Returns:
        torch.Tensor or np.ndarray: Normalized image with nonzero values scaled to [0,1].
    """
    is_tensor = isinstance(image, torch.Tensor)
    # print("is tensor", is_tensor)
    
    if not (is_tensor or isinstance(image, np.ndarray)):
        raise TypeError("Input must be a PyTorch tensor or a NumPy array.")
    
    nonzero_mask = image != 0  # Boolean mask for nonzero elements
    
    if is_tensor:
        if torch.any(nonzero_mask):  # Ensure there are nonzero values
            image_min = torch.min(image[nonzero_mask])
            image_max = torch.max(image[nonzero_mask])

            if not (image_min >= 0 and image_max <= 1):  # Only normalize if needed
                image[nonzero_mask] = (image[nonzero_mask] - image_min) / (image_max - image_min)
    
    else:  # NumPy array case
        if np.any(nonzero_mask):  # Ensure there are nonzero values
            image_min = np.min(image[nonzero_mask])
            image_max = np.max(image[nonzero_mask])

            if not (image_min >= 0 and image_max <= 1):  # Only normalize if needed
                image[nonzero_mask] = (image[nonzero_mask] - image_min) / (image_max - image_min)
    
    return image

def copy_files(src_folder, dst_folder):
    """
    Copies all files from src_folder to dst_folder.
    
    Args:
        src_folder (str): Source directory path.
        dst_folder (str): Destination directory path.
    """
    # Ensure the destination folder exists
    os.makedirs(dst_folder, exist_ok=True)

    # Loop through all files in the source directory
    for count, filename in enumerate(os.listdir(src_folder)):
        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)

        # Check if it's a file before copying
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)  # copy2 preserves metadata

    print(f"Copied {count} ship images to {dst_folder}.")

    # print(f"All files copied from {src_folder} to {dst_folder}")

