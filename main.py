from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
import numpy as np
import base64
import cv2
import uvicorn
from PIL import Image
import os
import shutil
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)


def apply_brightness_contrast(input_img, brightness=255, contrast=127):
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    cv2.putText(buf, 'B:{},C:{}'.format(brightness, contrast), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return buf


def map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


app = FastAPI()
refine_net = load_refinenet_model(cuda=False)
craft_net = load_craftnet_model(cuda=False)

def extractor(image, original_image):

    # read image
    image = read_image(image)

    # load models


    # perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.5,
        link_threshold=0.4,
        low_text=0.2,
        cuda=False,
        long_size=1880
    )
    output_dir = "./output/"
    image = original_image
    # export detected text regions
    exported_file_paths = export_detected_regions(
        image=image,
        regions=prediction_result["boxes"],
        output_dir=output_dir,
        rectify=True
    )
    return exported_file_paths

def ocr(dirs):
    Res=[]
    for z,image in enumerate(dirs):
        Res.append(pytesseract.image_to_string(Image.open(image), lang='fas',config='--psm 7 --oem 3'))
    return Res
# unload models from gpu
empty_cuda_cache()


@app.get("/")
async def root():
    return {"message": "Hello World"}


class Analyzer(BaseModel):
    encoded_img: bytes

@app.post("/analyze")
async def analyze_route(file: Analyzer):
    nparr = np.fromstring(base64.b64decode(file.encoded_img), np.uint8)
    org_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = apply_brightness_contrast(org_img, contrast=254, brightness=300)
    dirs = extractor(img, org_img)
    text = ocr(dirs)
    os.system("rm -r ./output")
    return {"text":text}


@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    contents = await uploaded_file.read()
    nparr = np.fromstring(contents, np.uint8)
    org_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = apply_brightness_contrast(org_img, contrast=254, brightness=300)
    dirs = extractor(img, org_img)
    text = ocr(dirs)
    os.system("rm -r ./output")
    return {"text": text}


@app.post('/upload')
async def upload_file(file: UploadFile=File(...)):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    return {
        "filename": file.filename,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
