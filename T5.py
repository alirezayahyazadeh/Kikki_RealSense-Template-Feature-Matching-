import cv2
import numpy as np
from PIL import Image
from rembg import remove

def remove_background(input_image_path, output_image_path):
    """
    Removes the background from an RGB image using rembg.
    Saves the result as a PNG with transparency.
    """
    # Load the input image using Pillow
    input_image = Image.open(input_image_path).convert("RGB")
    
    # Use rembg to remove the background
    output_image = remove(input_image)
    
    # Save the result (PNG with alpha channel)
    output_image.save(output_image_path)

def main():
    # Paths to input/output
    input_path = r"D:\template.jpg"
    output_path = r"D:\output_no_bg.png"
    
    # Remove background
    remove_background(input_path, output_path)
    
    # Optionally, display the result using OpenCV
    # Note that the output is RGBA (with alpha channel)
    rgba_image = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
    
    # Convert RGBA to BGRA for display in OpenCV if needed
    # (OpenCV uses BGR or BGRA channel ordering)
    # If the .png has an alpha channel, it should be read as BGRA automatically.
    
    # Show the result in a window
    cv2.imshow("No Background", rgba_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
