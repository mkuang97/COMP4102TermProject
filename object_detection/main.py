import typing
import numpy as np
import cv2
import imutils
from scipy import ndimage
from skimage.feature import peak_local_max
from torchvision import transforms
import random
import torch
import json
# from skimage.morphology import watershed

# from my_utility import Watershed
import torchvision.models as models
import os

dirname = os.path.dirname(__file__)

IMAGE_ROOT = "../images/"

def get_coco_object_categories():
    categories = {}
    filename = os.path.join(dirname, "annotations/instances_val2017.json")
    with open(filename,'r') as COCO:
        js = json.loads(COCO.read())
    for cat in js['categories']:
        categories[cat['id']] = cat['name']
    return categories

def basic_segmentation(img):
    new_img = img.copy()
    shifted = cv2.pyrMeanShiftFiltering(img, 21,51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # loop over the contours
    for (i, c) in enumerate(contours):
        # draw the contour
        ((x, y), _) = cv2.minEnclosingCircle(c)
        cv2.putText(new_img, "#{}".format(i + 1), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.drawContours(new_img, [c], -1, (0, 255, 0), 2)
    
    cv2.imshow("Shifted", shifted)
    cv2.imshow("Thresh", thresh)
    return new_img

def img_to_tensor(img):
    h, w, c = img.shape
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    tensor_img = img_transform(img)
    batch = tensor_img.unsqueeze(0)
    return batch

def display_tensor_img(tensor_img):
    img_transform = transforms.Compose([
        transforms.ToPILImage()
    ])
    pil_img = img_transform(tensor_img)
    return pil_img

    
def retrieve_detected_objects(img):
    '''
    We assume the image is an RGB image, We use a pretrained maskrcnn model on the coco dataset
    '''
    rn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    rn.eval()
    tensor_img = img_to_tensor(img)
    predictions = rn(tensor_img)[0]
    masks = predictions['masks'].detach().numpy()
    labels = predictions['labels'].detach().numpy()
    boxes = predictions['boxes'].detach().numpy()
    scores = predictions['scores'].detach().numpy()
    coco_cat = get_coco_object_categories()
    colors = []

    detected_masks = []
    detected_labels = []
    det_img = np.copy(img)
    for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
        if score > 0.50:
            print('drawing label {} '.format(coco_cat[label]))
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            colors.append(color)
            det_img = cv2.rectangle(det_img, (box[0],box[1]), (box[2],box[3]), color=color, thickness=1)
            c, h, w = mask.shape
            mask = mask.reshape(h, w, c)
            detected_masks.append(mask)
            detected_labels.append(coco_cat[label])
            # Color in mask
            for i in range(det_img.shape[0]):
                for j in range(det_img.shape[1]):
                    if mask[i][j] > 0.50:
                        det_img[i][j] = color
            # Put label
            det_img = cv2.putText(det_img, "{} - {:.2f}".format(coco_cat[label], score), (box[0],int(box[1]+10)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=1)
    # cv2.imshow('Detected Objects', img)
    # print("masks:", masks)
    return det_img, detected_masks, detected_labels

def main():
    img = cv2.imread(IMAGE_ROOT+"person_dog.jpg", cv2.IMREAD_COLOR)

    # watershed = Watershed(img)
    # watershed.segment2()
    # b = basic_segmentation(img)
    # cv2.imshow("Basic Segmentation", b)
    # test_watershed = my_watershed(img)
    # cv2.imshow("Water Shed result", test_watershed)
    # cv2.waitKey()
    # dst = watershed1(img)

    masks, labels = retrieve_detected_objects(img)
    cv2.waitKey()
if __name__=='__main__':
    main()