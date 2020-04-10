import argparse
import cv2
from object_detection.main import retrieve_detected_objects
from object_detection.utility import Watershed
from matplotlib import pyplot as plt
import numpy as np
from exemplar_based_inpainting.source.InpainterV2 import InpainterV2 as paint
import sys

FLAGS = None

def rescale(img, maxh=512, maxw=512):
    h, w, c = img.shape
    max_height, max_width = maxh, maxw
    ratio = min(max_height/h, max_width/w)
    new_h, new_w = int(h*ratio), int(w*ratio)
    return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
def main():
    img = cv2.imread(FLAGS.image_file, cv2.IMREAD_COLOR)

    print("Detecting objects from input image...")
    det_image, masks, labels = retrieve_detected_objects(img)
    
    # resize the image to a workable size (we want to )
    img = rescale(img)
    height, width, channel = img.shape
    det_img = rescale(det_image)

    print("Labels: {}".format(labels))
    print("Retreived masks and labels...")
    # Convert masks to uint8 type and binarize it (either 0 or 255)
    for i, mask in enumerate(masks):
        masks[i][masks[i]  > 0.25] = 255
        masks[i] = masks[i].astype(np.uint8)
        masks[i] = rescale(masks[i])

    det_image = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
    plt.imshow(det_image)
    plt.show(block=False)
    while True:
        prompt = "Select which mask to infill (-1 to exit): \n"
        for i , label in enumerate(labels):
            prompt += "{}. {}\n".format(i, label)
        selected = int(input(prompt))
        if selected == -1:
            break
        if selected < 0 or selected >= len(labels):
            print("Selected number is out of bounds")
            continue
        print("Selected to inpaint {} mask".format(labels[selected]))
        halfPatchWidth = 4
        i = paint(img, masks[selected].reshape(height, width), halfPatchWidth)
        if i.checkValidInputs() == True:
            i.doInpaint()
            cv2.imwrite("./results/result.jpg", i.result)
            plt.imshow(i.result)
            plt.show(block=False)
        else:
            print("invalid dimensions")
            print("img shape {}, dtype {}".format(img.shape, img.dtype))
            print("mask shape {}, dtype {}".format(masks[selected].shape, masks[selected].dtype))
            # print(masks[selected].shape)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file", type=str)
    FLAGS = parser.parse_args()
    main()