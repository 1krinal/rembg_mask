import cv2
import urllib.request
import numpy as np

def capture_Event(event, x, y, flags, params):
    # If the left mouse button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the coordinate of the clicked point
        print(f"({x}, {y})")

if __name__ == "__main__":
    image_url = "https://upload.wikimedia.org/wikipedia/commons/2/2a/Junonia_lemonias_DSF_upper_by_Kadavoor.JPG"

    # Download the image from the URL and save it locally
    urllib.request.urlretrieve(image_url, 'downloaded_image.jpg')

    # Read the downloaded image
    img = cv2.imread('downloaded_image.jpg', 1)

    # Show the Image
    cv2.imshow('image', img)

    # Set the Mouse Callback function, and call
    # the Capture_Event function.
    cv2.setMouseCallback('image', capture_Event)

    # Press any key to exit
    cv2.waitKey(0)