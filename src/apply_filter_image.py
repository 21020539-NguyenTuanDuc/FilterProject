import csv
from typing import List, Tuple

import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from PIL import Image, ImageDraw
import torch
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
import numpy as np
from src.models.dlib_module import DlibLitModule
from src.models.components.simple_regnet import SimpleRegnet
import cv2;
import torchvision
from scipy.spatial import Delaunay
from operator import itemgetter

import src.filter_function as ff
import src.faceBlendCommon as fbc

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.dlib_datamodule import TransformDataset  # noqa: E402
from src import utils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig):

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    metric_dict = trainer.callback_metrics
    model = torch.load(cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    log.info("Starting predictions!")

    filter = ff.read_filter("FilterImage/StoneMask")
    filter_img = filter["img"]
    b, g, r, filter_alpha = cv2.split(filter_img)
    filter_img = cv2.merge((b, g, r))
    filter_lm = filter["points"]
    filter_hull = filter["hull"]
    filter_hullIndex = filter["hullIndex"]
    filter_size = filter_img.shape
    filter_rect = (0, 0, filter_size[1], filter_size[0])
    filter_tri = filter["tri"]
    
    frame = cv2.imread("testImage/w_face_test.jpg")
    warped_img = np.copy(frame)
    image_lm = get_image_lm("testImage/w_face_test.jpg", model)

    image_hull = []
    for i in range(0, len(filter_hullIndex)):
        image_hull.append(image_lm[filter_hullIndex[i][0]])

    mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
    mask1 = cv2.merge((mask1, mask1, mask1))
    filter_alpha_mask = cv2.merge((filter_alpha, filter_alpha, filter_alpha))

    for i in range(len(filter_tri)):
        t1 = []
        t2 = []
        
        for j in range(3):
            t1.append(filter_hull[filter_tri[i][j]])
            t2.append(image_hull[filter_tri[i][j]])
        
        fbc.warpTriangle(filter_img, warped_img, t1, t2)
        fbc.warpTriangle(filter_alpha_mask, mask1, t1, t2)

    # Blur the mask before blending
    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

    mask2 = (255.0, 255.0, 255.0) - mask1

    # Perform alpha blending of the two images
    temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
    output = temp1 + temp2

    frame = output = np.uint8(output)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(np.uint8(rgb_image))
    pil_image.save("FilterImage/result.png")

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)

def get_image_lm(image_path, model):

    result_lm = []

    imgBB = cv2.imread(image_path, cv2.IMREAD_COLOR)
    (h, w) = imgBB.shape[:2]

    face_detector = cv2.dnn.readNetFromCaffe("BBDetection\deploy.prototxt.txt"
                                             , "BBDetection/res10_300x300_ssd_iter_140000.caffemodel")
    blob = cv2.dnn.blobFromImage(cv2.resize(imgBB, (300, 300)), 1.0, (300, 300))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    startX, startY, endX, endY = 0, 0, 0, 0
    max_confidence = -1
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        max_confidence = max(max_confidence, confidence)

        # Filter out weak detections
        if confidence > 0.1 and confidence == max_confidence:
            # Get the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
            (tempStartX, tempStartY, tempEndX, tempEndY) = box.astype("int")
            if(tempStartX >= 0 and tempStartY >= 0 and tempEndY <= h and tempEndX <= w):
                startX = int(1.0 * tempStartX * w / 300)
                startY = int(1.0 * tempStartY * h /300)
                endX = int(1.0 * tempEndX * w / 300)
                endY = int(1.0 * tempEndY * h /300)

    # cv2.rectangle(imgBB, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # cv2.imwrite("testImage/test_BB.png", imgBB)

    imgBB = imgBB[startY:endY, startX:endX] # crop image
    bb_h = endY - startY
    bb_w = endX - startX
    if imgBB.size == 0:
        print("Cropped error!")
        color_converted_img = cv2.cvtColor(imgBB, cv2.COLOR_BGR2RGB)
    else:
        color_converted_img = cv2.cvtColor(imgBB, cv2.COLOR_BGR2RGB)
    transform = Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    transformB = Compose([
        ToTensorV2(),])
    input_image = Image.fromarray(color_converted_img)
    input_image = np.asarray(input_image)
    origin_input_tensor = transformB(image=np.array(Image.open(image_path).convert('RGB')))
    origin_input_tensor = origin_input_tensor["image"].unsqueeze(0)
    input_tensor = transform(image=input_image)
    input_tensor = input_tensor['image'].unsqueeze(0)
    with torch.no_grad():
        landmarks = model(input_tensor)
    landmarks = landmarks.squeeze().numpy()
    for lm in landmarks:
        lm = (lm + 0.5) * np.array([bb_w, bb_h]) # convert to image pixel coordinates
        lm = lm + np.array([startX, startY])
        result_lm.append([int(lm[0]), int(lm[1])])

    face_padding = 100 # padding for forehead
    result_lm.append([result_lm[0][0], startY + face_padding])
    result_lm.append([result_lm[16][0], startY + face_padding])

    return np.array(result_lm)
    
if __name__ == "__main__":
    main()