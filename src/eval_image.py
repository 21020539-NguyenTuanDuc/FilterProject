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

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.dlib_datamodule import TransformDataset  # noqa: E402
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig):
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

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

    annotated_image = eval_image(model=model, image_path='testImage/image_0014.png')
    # annotated_image = eval_image(model=model, image_path='D:\AI\datasets\FilterProject/testImage/test.jpg')
    torchvision.utils.save_image(annotated_image, "testImage/frame1_res.png")
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)

def eval_image(image_path, model):
    # Make bounding box
    imgBB = cv2.imread(image_path)
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
        if confidence > 0.3 and confidence == max_confidence:
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
    # cv2.imshow("Output", imgBB) #To run in Google Colab, comment out this line Colab notebook
    # #cv2_imshow(image) #To run in Google Colab, uncomment this line
    # cv2.waitKey(0)
    if imgBB.size == 0:
        print("Cropped error!")
        print(f"(w, h) {(w, h)}, (startX, startY, endX, endY) {startX, startY, endX, endY}")
        color_converted_img = cv2.cvtColor(imgBB, cv2.COLOR_BGR2RGB)
    else:
        print(f"(w, h) {(w, h)}, (startX, startY, endX, endY) {startX, startY, endX, endY}")
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
        output_tensor = model(input_tensor)
    print(input_tensor.shape, output_tensor.shape) #1 3 224 224, 1 68 2
    annotated_image = annotate_original_tensor(origin_input_tensor, output_tensor, startX, startY, bb_w, bb_h)
    # annotated_image = TransformDataset.annotate_tensor(input_tensor, output_tensor)
    return annotated_image

def annotate_original_tensor(image: torch.Tensor, landmarks: np.ndarray, startX: int, startY: int, bb_w, bb_h) -> Image:
    transform_to_img = torchvision.transforms.ToPILImage()
    images = image
    images_to_save = []
    for lm, img in zip(landmarks, images):
        new_img = transform_to_img(img)
        img = img.cpu().numpy()
        lm = lm.to('cpu').numpy()
        lm = (lm + 0.5) * np.array([bb_w, bb_h]) # convert to image pixel coordinates
        lm = lm + np.array([startX, startY])
        img = annotate_original_image(new_img, lm)
        images_to_save.append( torchvision.transforms.ToTensor()(img) )

    return torch.stack(images_to_save)

def annotate_original_image(image: Image, landmarks: np.ndarray) -> Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw_radius = int(1.0 * width/680)
    for i in range(landmarks.shape[0]):
        draw.ellipse((landmarks[i, 0] - draw_radius, landmarks[i, 1] - draw_radius,
                        landmarks[i, 0] + draw_radius, landmarks[i, 1] + draw_radius), fill=(255, 255, 0))
    return image
if __name__ == "__main__":
    main()