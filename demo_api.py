from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import FileResponse
import cv2
import uuid
import os
import aiohttp
import numpy as np
import rembg
from pathlib import Path

ORIGINAL ="original/"
MASK ="masked/"
app = FastAPI()


async def download_image_and_process(image_url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            contents = await response.read()

    filename = f"{uuid.uuid4()}.jpg"
    with open(f"{ORIGINAL}{filename}", "wb") as f:
        f.write(contents)
    output_data = rembg.remove(contents)  

    with open(f"{MASK}mask_{filename}", "wb") as output_file:
        output_file.write(output_data)

    mask_image = cv2.imdecode(np.frombuffer(output_data, np.uint8), cv2.IMREAD_UNCHANGED)
    mask_image[mask_image > 0] = 255  # Set non-zero pixels to 255 (white)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    mask_filename = f"mask_BW_{filename}"
    cv2.imwrite(f"{MASK}{mask_filename}", mask_image)
            
    return {"filename": filename, "mask_filename": mask_filename,"output_file_path": f"mask_{filename}"}

def get_image_path(filename: str, mask: bool = False) -> Path:
    if  mask==True:
        return Path(MASK)/ f"mask_{filename}"
    else:
        return Path(MASK)/f"mask_BW_{filename}"
    
@app.post("/image_url/")
async def remove_background(image_url: str):
    filename = await download_image_and_process(image_url)
    return {"image": filename}


@app.get("/image/{filename}")
async def get_object_detect_image(filename: str, mask: bool = False): 
    image_path = get_image_path(filename, mask)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)
    
    
@app.post("/merge_images/")
async def merge_images(original_filename: str, file: UploadFile = File(...)):
   
    original_path = os.path.join(ORIGINAL, original_filename)
    original_image = cv2.imread(original_path)

    # Load the original image
    filename = file.filename
    mask_path = os.path.join(MASK, filename)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    merged_image = cv2.bitwise_and(original_image, original_image, mask=mask_image)

    # Save the merged image
    merged_filename = f"merged_{filename}"
    merged_path = os.path.join(ORIGINAL, merged_filename)
    cv2.imwrite(merged_path, merged_image)

    return {"original Image":original_path,"mask image":mask_path,"merged_filename": merged_filename}

def get_mereg_image(filename: str,mask:bool=False ) -> Path:
    if  mask:
        return Path(ORIGINAL)/f"merged_mask_BW_{filename}"
    else:
        return Path(MASK)/f"mask_BW_{filename}"

@app.get("/merge_image/{filename}")
async def get_image(filename: str,mask:bool=False ):
    
    path = get_mereg_image(filename,mask)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)
