# Python program to explain 
# mask inversion on a b/w image. 

import numpy as np
import cv2
import urllib.request

# Function to load an image from a URL
def load_image_from_url(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

# Image URL
image_url = 'https://upload.wikimedia.org/wikipedia/commons/2/2a/Junonia_lemonias_DSF_upper_by_Kadavoor.JPG'

# Load the image from the URL
img = load_image_from_url(image_url)

# Rest of your code...
pts = np.array([[10, 150], [150, 100], [300, 150], [350, 100], [310, 20], [35, 10]])

# Make mask
pts = pts - pts.min(axis=0)

mask = np.zeros(img.shape[:2], np.uint8)
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

# Do bit-op
dst = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite("mask.png", mask)
cv2.imwrite("dst.png", dst)
















