'''
k-means object detection
'''
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from main import retrieve_detected_objects

figure_size = 10

def kmeans(image, k):
    # Need to convert from RGB to HSV; 
    # color descriptions in terms of hue/lightness/saturation are more relevant
    converted_color_img = cv2.cvtColor(cv2.blur(image,(10,10)),cv2.COLOR_BGR2RGB)

    vectorized = converted_color_img.reshape((-1,3))
    vectorized = np.float32(vectorized)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    attempts=10
    ret,label,center=cv2.kmeans(vectorized, k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]
    # apply kmeans clustering to original image
    res2 = res.reshape((converted_color_img.shape))

    _, thresh = cv2.threshold(cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY), 110, 255, 0) 
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    major_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 35:
            major_contours.append(contour)
    
    mask = np.zeros((image.shape))
    mask[:] = (255, 255, 255)
    cv2.fillPoly(mask, pts =major_contours, color=(0,0,0))

    cv2.imshow("kmeans with k = " + str(k), mask)
    cv2.waitKey(0)

def canny(image):    
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

    edges = cv2.Canny(image,100,200)

    _, contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    contoured_img = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    mask = np.zeros((image.shape))
    mask[:] = (255, 255, 255)
    cv2.fillPoly(mask, pts =contours, color=(0,0,0))

    cv2.imshow("canny", mask)
    cv2.waitKey(0)

def tests():
    
    IMAGE_ROOT = "../images/"
    img = cv2.imread(IMAGE_ROOT+"cards.jpg", cv2.IMREAD_COLOR)

    k = 7
    kmeans(img, k)
    # canny(img)

    # cv2.imshow("original img", img)
    # cv2.waitKey(0)
    # exit()

    # def rescale(img, maxh=512, maxw=512):
    #     h, w, c = img.shape
    #     max_height, max_width = maxh, maxw
    #     ratio = min(max_height/h, max_width/w)
    #     new_h, new_w = int(h*ratio), int(w*ratio)
    #     return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # print("Detecting objects from input image...")
    # det_image, masks, labels = retrieve_detected_objects(img)

    # # resize the image to a workable size (we want to )
    # img = rescale(img)
    # height, width, channel = img.shape
    # det_img = rescale(det_image)

    # print("Labels: {}".format(labels))
    # print("Retreived masks and labels...")
    # # Convert masks to uint8 type and binarize it (either 0 or 255)
    # for i, mask in enumerate(masks):
    #     print('masks[i] is', masks[i])
    #     masks[i][masks[i]  > 0.25] = 255
    #     masks[i] = masks[i].astype(np.uint8)
    #     masks[i] = rescale(masks[i])
    #     print('masks[i] is', masks[i])

    # det_image = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
    # plt.imshow(det_image)
    # plt.show(block=False)

if __name__ == '__main__':
    tests()

   