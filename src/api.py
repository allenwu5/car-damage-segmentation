import os
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import ORJSONResponse

from model import predict
assert torch.cuda.is_available()
from utils import binary_mask_to_rle
import base64
from pycocotools import mask
app = FastAPI()

def ocr_predict(file_content, file_name):
    vis_output, outputs = predict(file_content)

    
    _, encoded_img = cv2.imencode('.PNG', vis_output.get_image()[:, :, ::-1])

    encoded_img = base64.b64encode(encoded_img)

    images = []
    for key, value in outputs.items():
        if key == 'instances':
            scores = value.get('scores').tolist()
            pred_classes = value.get('pred_classes').tolist()
            pred_masks = value.get('pred_masks').detach().cpu().numpy()
            for i in range(len(scores)):
                pred_masks_rle = binary_mask_to_rle(np.asfortranarray(pred_masks[i]))
                images.append({
                    'score': scores[i],
                    'pred_class': pred_classes[i],
                    'pred_masks_rle': pred_masks_rle
                })
    return {
        'file_name': file_name,
        'encoded_img': encoded_img,
        'images': images
    }


@app.post("/ocr/predict", response_class=ORJSONResponse)
async def ocr_predict_api(file: UploadFile = File(...)):
    # https://stackoverflow.com/questions/65350640/to-upload-a-file-what-are-the-pros-and-cons-of-sending-base64-in-post-body-vs-mu
    prediction = ocr_predict(await file.read(), file.filename)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
