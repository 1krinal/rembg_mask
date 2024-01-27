
import rembg
import cv2
import numpy as np
import urllib.request

def remove_background(input_url, output_path, mask_path):
    # Download the image from the URL
    input_data = urllib.request.urlopen(input_url).read()

    # Use rembg to remove the background
    output_data = rembg.remove(input_data)

    # Save the image with the removed background
    with open(output_path, "wb") as output_file:
        output_file.write(output_data)

    # Create a mask from the image with the removed background
    create_mask(output_path, mask_path)

def create_mask(input_path, mask_path):
    # Read the image with the removed background
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Extract the alpha channel (transparency) from the image
    alpha_channel = image[:, :, 3]

    # Create a black and white mask by thresholding the alpha channel
    _, mask = cv2.threshold(alpha_channel, 128, 255, cv2.THRESH_BINARY)

    # Save the mask image
    cv2.imwrite(mask_path, mask)


# Example usage
input_url = "https://upload.wikimedia.org/wikipedia/commons/2/2a/Junonia_lemonias_DSF_upper_by_Kadavoor.JPG"
output_image_path = "output_image_without_background.png"
mask_image_path = "mask_image.png"

remove_background(input_url, output_image_path, mask_image_path)