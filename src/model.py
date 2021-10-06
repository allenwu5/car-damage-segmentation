import os

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

val_metadata_dicts = MetadataCatalog.get("car_dataset_val")
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# faster, and good enough for this dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# set a custom testing threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.DATASETS.TEST = ("car_dataset_val", )
predictor = DefaultPredictor(cfg)


def predict(file_content):
    np_array = np.fromstring(file_content, np.uint8)
    im = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1],
                   metadata=val_metadata_dicts,
                   scale=0.5,
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   instance_mode=ColorMode.IMAGE_BW
                   )
    vis_output = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return vis_output, outputs
