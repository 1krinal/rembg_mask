from rembg import remove
import requests
from PIL import Image
from io import BytesIO
import os
import cv2
import numpy as np
import urllib.request

def remove_background_rembg(input_url, output_path):
    # Download the image from the URL
    response = requests.get(input_url)
    img = Image.open(BytesIO(response.content))
    img.save('original/input_image.jpg', format='jpeg')

    # Use rembg to remove background
    input_data = open('original/input_image.jpg', 'rb').read()
    with open(output_path, 'wb') as o:
        re = remove(input_data)
        o.write(re)

def remove_background_polygon(input_url, output_path):
    # Download the image from the URL
    response = urllib.request.urlopen(input_url)
    image_data = response.read()
    with open('original/input_image.png', 'wb') as f:
        f.write(image_data)

    # Load the image
    image = cv2.imread('original/input_image.png', cv2.IMREAD_UNCHANGED)

    # Define the polygon points
    points = np.array([(378, 167)
    ,(412, 184)
    ,(446, 143)
    ,(403, 132)
    ,(399, 67)
    ,(401, 43)
    ,(430, 61)
    ,(410, 77)
    ,(392, 64)], dtype=np.int32)

    # Create a mask using the polygon points
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(output_path, result)


def merge_images(revbg_path, polygon_path, output_path):
    # Load images
    rembg_output = Image.open(revbg_path)
    polygon_output = Image.open(polygon_path)

    # Convert PIL images to numpy arrays
    rembg_np = np.array(rembg_output)
    polygon_np = np.array(polygon_output)

    # Alpha blend the images
    alpha_blend = cv2.addWeighted(rembg_np, 1, polygon_np, 0.5, 0)

    # Convert the result back to PIL Image
    result_image = Image.fromarray(alpha_blend)

    # Save the merged image
    result_image.save(output_path)  

if __name__ == "__main__":
    input_url = r"C:\Users\Shree\core\prectical\original\image.jpg"
    output_path_rembg = 'masked/rembg_output.png'
    output_path_polygon = 'masked/polygon_output.png'
    merged_output_path = 'masked/merged_output.png'

    # Merge images
    merge_images(output_path_rembg, output_path_polygon, merged_output_path)

    print("Images merged successfully.")

