# USAGE
# python predict.py

# import the necessary packages
from model import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import onnxruntime as ort

from PyQt5.QtCore import QLibraryInfo


def get_img_name(imagePath):
    imagePathR = "".join(reversed(imagePath))
    pos = imagePathR.find("/")
    return imagePath[len(imagePath) - pos : len(imagePath) - 4]


def prepare_plot(origImage, origMask, predMask, predProb, imagePath, suffix=""):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axs = axs.flatten()

    axs[0].imshow(origImage)
    axs[1].imshow(origMask)
    axs[2].imshow(predMask)
    axs[3].imshow(predProb)

    axs[0].set_title("Image")
    axs[1].set_title("Original Mask")
    axs[2].set_title(f"Predicted Mask ({suffix})")
    axs[3].set_title(f"Predicted Probability ({suffix})")

    fig.tight_layout()
    filename = "predict_plot_" + suffix + "_" + get_img_name(imagePath)
    plotPath = config.PLOT_PATH.replace(get_img_name(config.PLOT_PATH), filename)
    
    fig.savefig(plotPath)
    plt.close(fig)
    print(f"[INFO] Saved plot to {plotPath}")

def preprocess_image(imagePath):
    """Préparation commune de l'image (chargement, resize, normalisation)"""
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    
    image = cv2.resize(image, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))
    orig = image.copy()
    
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    
    filename = imagePath.split(os.path.sep)[-1]
    groundTruthPath = os.path.join(config.MASKS_PATH, filename)
    gtMask = cv2.imread(groundTruthPath, 0)
    gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))
    
    return image, orig, gtMask

def make_predictions_pytorch(model, imagePath):
    image_np, orig, gtMask = preprocess_image(imagePath)
    image_tensor = torch.from_numpy(image_np).to(config.DEVICE)
    model.eval()
    with torch.no_grad():
        predMask = model(image_tensor).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()

        predProb = predMask * 255
        predMask = (predMask > config.THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)

        prepare_plot(orig, gtMask, predMask, predProb, imagePath, suffix="PyTorch")

def make_predictions_onnx(session, imagePath):
    image_np, orig, gtMask = preprocess_image(imagePath)
    input_name = session.get_inputs()[0].name

    ort_outputs = session.run(None, {input_name: image_np})
    predMask = ort_outputs[0].squeeze()
    predMask = 1 / (1 + np.exp(-predMask))
    
    predProb = predMask * 255
    predMask = (predMask > config.THRESHOLD) * 255
    predMask = predMask.astype(np.uint8)

    prepare_plot(orig, gtMask, predMask, predProb, imagePath, suffix="ONNX")


if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
        QLibraryInfo.PluginsPath
    )

    # load the image paths in our testing file and randomly select 10
    # image paths
    print("[INFO] loading up test image paths...")
    imagePaths = open(config.TEST_PATH).read().strip().split("\n")
    imagePaths = np.random.choice(imagePaths, size=10)

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")
    unet = torch.load(config.BEST_MODEL_PATH, weights_only=False).to(config.DEVICE)
    
    onnx_path = os.path.join(config.BASE_OUTPUT, "model.onnx")
    ort_session = ort.InferenceSession(onnx_path)

    # iterate over the randomly selected test image paths
    for path in imagePaths:
        # make predictions and visualize the results
        make_predictions(unet, path)
        make_predictions_onnx(onnx_model, path)
    
