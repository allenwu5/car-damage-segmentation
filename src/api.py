import base64

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import ORJSONResponse
from label_studio_converter.brush import encode_rle

from model import predict

assert torch.cuda.is_available()

app = FastAPI()


def predict_container_damage(file_content, file_name):
    vis_output, outputs = predict(file_content)
    _, encoded_img = cv2.imencode('.PNG', vis_output.get_image()[:, :, ::-1])
    encoded_img = base64.b64encode(encoded_img)

    # 
    results = []
    for key, value in outputs.items():
        if key == 'instances':
            scores = value.get('scores').tolist()
            pred_classes = value.get('pred_classes').tolist()
            pred_masks = value.get('pred_masks').detach().cpu().numpy()

            for i in range(len(scores)):
                mask = pred_masks[i].astype(int) * 255
                mask = np.dstack((mask, mask, mask, mask))
                pred_masks_rle = encode_rle(mask.flatten())

                results.append({
                    'score': scores[i],
                    'pred_class': pred_classes[i],
                    'pred_masks_rle': pred_masks_rle
                })
    return {
        'file_name': file_name,
        'encoded_img': encoded_img,
        'results': results
    }


@app.post("/container/damage", response_class=ORJSONResponse)
async def predict_container_damage_api(file: UploadFile = File(...)):
    prediction = predict_container_damage(await file.read(), file.filename)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
