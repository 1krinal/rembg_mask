import rembg
import cv2
import numpy as np
import urllib.request
from PIL import Image
 

def load_image_from_url(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def remove_background_and_create_mask(input_url, output_image_path, mask_path):
    # Download the image from the URL
    input_data = urllib.request.urlopen(input_url).read()
    output_data = rembg.remove(input_data)

    # Save the image with the removed background
    with open(output_image_path, "wb") as output_file:
        output_file.write(output_data)

    # Create a mask using contour-based approach
    create_mask(output_image_path, mask_path)

def create_mask(input_path, mask_path):
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Extract the alpha channel (transparency) from the image
    alpha_channel = image[:, :, 3]

    # Create a black and white mask by thresholding the alpha channel
    _, mask = cv2.threshold(alpha_channel, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite(mask_path, mask)

# Example usage
input_url = "https://images.all-free-download.com/images/graphiclarge/swan_beauty_nature_264533.jpg"
# urllib.request.urlretrieve(input_url, 'original/downloaded_image.png')
cv2.imread('original/downloaded_image.png', 1)

output_image_path = "original/output_image_without_background1.png"
mask_image_path = "masked/mask_image1.png"

remove_background_and_create_mask(input_url, output_image_path, mask_image_path)

img = load_image_from_url(input_url)
# pts = np.array([(196, 202),(186, 233),(192, 245),(202, 238),(219, 195),(206, 187),(196,202)])

p1 = np.array([(47, 151), (27, 235), (98, 269), (140, 177), (47, 151)], dtype=np.int32)
p2 = np.array([(45, 517), (23, 589), (129, 575), (142, 513), (45, 517)], dtype=np.int32)
p3 = np.array([(204, 189),(202, 199),(195, 201),(193, 212),(187, 224),(185, 236),(190, 243),(197, 245),(207, 232),(213, 213),(218, 203),(220, 191),(206, 188),(202, 199)], dtype=np.int32)
p4 = np.array([(455, 187),(451, 253),(497, 279),(528, 205),(458, 179)], dtype=np.int32)

# Make mask
# pts = pts - pts.min()
pts = [p1, p2,p3,p4]
                     
mask = np.zeros(img.shape[:2], np.uint8)
# cv2.drawContours(mask, [pts], 0, (255,255,255), -1, cv2.LINE_AA)
cv2.fillPoly(mask, pts, (255, 255, 255))
dst = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite("masked/mask1.png", mask)
cv2.imwrite("original/result_with_mask1.png", dst)

# subtract two masking
mask_image1 = cv2.imread("masked/mask_image1.png", cv2.IMREAD_GRAYSCALE)
mask1 = cv2.imread("masked/mask1.png", cv2.IMREAD_GRAYSCALE)
# merged_mask = cv2.subtract(mask_image1, mask1)
merged_mask = cv2.bitwise_xor(mask_image1, mask1)
cv2.imwrite("masked/merged_mask.png", merged_mask)

#final output
original_image = cv2.imread(r'C:\Users\Shree\core\prectical\original\downloaded_image.png')
merged_mask = cv2.imread(r'C:\Users\Shree\core\prectical\masked\merged_mask.png', cv2.IMREAD_GRAYSCALE)
merged_mask_color = cv2.cvtColor(merged_mask, cv2.COLOR_GRAY2BGR)
result_image = cv2.bitwise_and(original_image, merged_mask_color)

cv2.imwrite(r'C:\Users\Shree\core\prectical\final_color_image.png', result_image)


