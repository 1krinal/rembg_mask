from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.responses import FileResponse
import cv2
import uuid
import io
import os
import aiohttp
import numpy as np
import rembg
import pathlib as Path
 

ORIGINAL ="original/"
MASK ="masked/"
app = FastAPI()


async def download_image_and_process(image_url: str):
    # Fetch the file contents from the URL
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    contents = await response.read()

    # Generate a unique filename
            filename = f"{uuid.uuid4()}.jpg"

    # Save the original file
            with open(f"{ORIGINAL}{filename}", "wb") as f:
                 f.write(contents)

    # Process the image using rembg
            output_data = rembg.remove(contents)  

    # Save the processed image
            with open(f"{MASK}mask_{filename}", "wb") as output_file:
                 output_file.write(output_data)

            mask_image = cv2.imdecode(np.frombuffer(output_data, np.uint8), cv2.IMREAD_UNCHANGED)
            mask_image[mask_image > 0] = 255  # Set non-zero pixels to 255 (white)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    # Save the black and white mask
            mask_filename = f"mask_BW_{filename}"
            cv2.imwrite(f"{MASK}{mask_filename}", mask_image)

            return {"filename": filename, "mask_filename": mask_filename,"output_file_path": f"mask_{filename}"}


@app.post("/image_url/")

async def remove_background(image_url: str):
     
    filename = await download_image_and_process(image_url)
    return {"image": filename}

@app.post("/merge_images/")
async def merge_images(original_filename: str, file: UploadFile = File(...)):
   
    original_path = os.path.join(ORIGINAL, original_filename)
    original_image = cv2.imread(original_path)

    # Load the original image
    filename = file.filename
    mask_path = os.path.join(MASK, filename)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Merge the two images
    merged_image = cv2.bitwise_and(original_image, original_image, mask=mask_image)

    # Save the merged image
    merged_filename = f"merged_{filename}"
    merged_path = os.path.join(ORIGINAL, merged_filename)
    cv2.imwrite(merged_path, merged_image)

    return {"original Image":original_path,"mask image":mask_path,"merged_filename": merged_filename}

@app.get("/show/")
async def read_random_file(original,mask):
 
    # get random file from the image directory
    file = os.listdir(ORIGINAL)
   
    filename = file.filename

    path = f"{ORIGINAL}{filename}"
     
    return FileResponse(path)
@app.get("/show/")
async def remove_background():
    
    filename = await download_image_and_process(image_url=f"{ORIGINAL/filename}")
    return {"image": filename}

    


 
# @app.get("/show/")
# async def read_random_file():
 
#     # get random file from the image directory
#     files = os.listdir(ORIGINAL)
#     random_index = randint(0, len(files) - 1)
 
#     path = f"{IMAGEDIR}{files[random_index]}"
     
#     return FileResponse(path)

# from fastapi import FastAPI, HTTPException
# import urllib.request
# import cv2
# import numpy as np
# import rembg
# import PIL as Image
# import io as BytesIO
# import requests

# app = FastAPI()

# def remove_background_and_create_mask(input_url: str, output_image_path:str, mask_path: str):
#     # Download the image from the URL
#     input_data = urllib.request.urlopen(input_url).read()
#     output_data = rembg.remove(input_data)

#     # Save the image with the removed background
#     with open(output_image_path, "wb") as output_file:
#         output_file.write(output_data)

#     # Create a mask using contour-based approach
#     create_mask(output_image_path, mask_path)

# def create_mask(input_path: str, mask_path: str):
#     image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

#     # Extract the alpha channel (transparency) from the image
#     alpha_channel = image[:, :, 3]

#     # Create a black and white mask by thresholding the alpha channel
#     _, mask = cv2.threshold(alpha_channel, 128, 255, cv2.THRESH_BINARY)
#     cv2.imwrite(mask_path, mask)

# @app.post("/remove_background_and_create_mask")
# async def remove_background_and_create_mask_endpoint(input_url: str, output_image_path: str, mask_path: str):
#     try:
#         response = requests.get(input_url)
#         response.raise_for_status()
#         image = Image.open(BytesIO(await response.buffer()))
#         remove_background_and_create_mask(input_url, output_image_path, mask_path)
#         return {"message": "Background removed and mask created successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/apply_custom_mask")
# async def apply_custom_mask(input_url: str, output_image_path: str, custom_mask_points: list):
#     try:
#         # Load the image from the URL
#         img_data = urllib.request.urlopen(input_url).read()
#         urllib.request.urlretrieve(img_data,'dowmloadimage.png')
#         nparr = np.frombuffer(img_data, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         # Create a black mask
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)

#         # Fill the mask with the custom point
#         pts = np.array(custom_mask_points, dtype=np.int32)
#         cv2.fillPoly(mask, [pts], (255, 255, 255))

#         # Apply the mask
#         masked_image = cv2.bitwise_and(img, img, mask=mask)

#         # Save the result image
#         cv2.imwrite(output_image_path, masked_image)

#         return {"message": "Custom mask applied successfully."}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
