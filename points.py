# import cv2
# import urllib.request
# import numpy as np

# def capture_Event(event, x, y, flags, params):
#     # If the left mouse button is pressed
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Print the coordinate of the clicked point
#         print(f"({x}, {y})")
         
      
# if __name__ == "__main__":
#     image_url = input("Enter user input:")

#     # Download the image from the URL and save it locally
#     urllib.request.urlretrieve(image_url, 'downloaded_image.png')

#     # Read the downloaded image
#     img = cv2.imread('downloaded_image.png', 1)


#     # Show the Image
#     cv2.imshow('image', img)

#     # Set the Mouse Callback function, and call
#     # the Capture_Event function.
#     cv2.setMouseCallback('image', capture_Event)

#     # Press any key to exit
#     cv2.waitKey(0)

from masking_image import mask_image_path 
import cv2


def capture_Event(event,x,y,flags,params):
    # If the left mouse button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the coordinate of the 
        # clicked point
        print(f"({x}, {y})")
if __name__=="__main__":
    # Read the Image.
    img = cv2.imread(mask_image_path)
    # Show the Image
    cv2.imshow('image', img)
    # Set the Mouse Callback function, and call
    # the Capture_Event function.
    cv2.setMouseCallback('image', capture_Event)
    # Press any key to exit
    cv2.waitKey(0)
    # Destroy all the windows
    cv2.destroyAllWindows()