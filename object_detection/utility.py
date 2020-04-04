import typing
import numpy as np
import cv2
import imutils
from scipy import ndimage
from skimage.feature import peak_local_max
import random
from scipy.ndimage import label

class Watershed:
    def __init__(self, image):
        self.image = image
        
        self.kernel = np.array([[1, 1, 1], 
                        [1, -8, 1], 
                        [1, 1, 1]], dtype=np.float32)

    def segment2(self, debug=True):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # First we obtain the borders/edges of our objects
        border = cv2.dilate(thresh, None, iterations=5)
        border = border - cv2.erode(border, None)

        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2 , 5)
        dist_transform = ((dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min()) * 255).astype(np.uint8)
        _, dist_transform = cv2.threshold(dist_transform, 180, 255, cv2.THRESH_BINARY)

        lbl, ncc = label(dist_transform)
        lbl = lbl*(255/(ncc+1))

        lbl[border == 255] = 255
        cv2.imshow("label0", lbl)
        lbl = lbl.astype(np.int32)
        cv2.watershed(self.image, lbl)
        lbl[lbl == -1] = 0
        lbl = lbl.astype(np.uint8)

        # print(dt)
        if debug:
            cv2.imshow("gray", gray)
            cv2.imshow("border", border)
            cv2.imshow("dt", dist_transform)
            cv2.imshow("label", lbl)

        x =  255 - lbl
        return 255 - lbl

        # gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray,(5,5), 0 )
        # # gray = cv2.Canny(self.image, 100, 200)
        # ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # # noise removal
        # kernel = np.ones((3,3),np.uint8)
        # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # # sure background area
        # sure_bg = cv2.dilate(opening,kernel,iterations=3)
        # cv2.imshow("opening", opening)
        # # Finding sure foreground area
        # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        # # Finding unknown region
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg,sure_fg)
        # # Marker labelling
        # ret, markers = cv2.connectedComponents(sure_fg)

        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers+1

        # cv2.imshow("marker_before", markers)
        # cv2.imshow("unknown", unknown)
        # # Now, mark the region of unknown with zero
        # markers[unknown==255] = 0
        # cv2.imshow("marker_after", markers)
        # markers = cv2.watershed(self.image,markers)

        # img = np.copy(self.image)
        # img[markers == -1] = [255,0,0]
        # cv2.imshow("markers", markers)
        # cv2.imshow("img after", img)
        # cv2.waitKey()

    def segment(self, debug=True):
        if debug:
            self.alpha_slider_max = 1000
            # perform a laplacian filtering
            #apply gaussian blur on top of the laplacian
            # self.image = cv2.GaussianBlur(self.image, (15,15), 0)
            cv2.imshow('Image with Black Background', self.image)
            self.imgLaplacian = cv2.filter2D(self.image, cv2.CV_32F, self.kernel)
            self.imgLaplacian = cv2.filter2D(self.image, cv2.CV_32F, self.kernel)
            self.sharp = np.float32(self.image)

            self.imgResult_int = self.sharp - self.imgLaplacian
            self.imgResult = np.clip(self.imgResult_int, 0, 255)
            self.imgResult = self.imgResult_int.astype('uint8')

            # self.imgLaplacian = np.clip(self.imgLaplacian, 0, 255)
            # self.imgLaplacian = np.uint8(self.imgLaplacian)
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Gray Image", self.gray)
            # _, self.gray_thresh = cv2.threshold(self.gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            _, self.gray_thresh = cv2.threshold(self.gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # cv2.imshow("Sharped Image", imgResult)
            # cv2.imshow("Laplacian after", imgLaplacian)
            cv2.namedWindow('Main Markers', cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Thresholding", "Main Markers" , 0, self.alpha_slider_max, lambda x: self.callback(x))

            cv2.waitKey()
        else:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # We assume that the background is white, and the objects we are trying to detect is darker
            ret, thresh = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            cv2.imshow("Threshold", thresh)
            # we need to remove noise so let's first erode the image, this way it removes edges and perhaps overlapping objects
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
            cv2.imshow("morphologyEx", opening)
            

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)
            cv2.imshow("sure_bg", sure_bg)
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5) 
            ret, sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)
            cv2.imshow("sure_fg", sure_fg)

            ## Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg,sure_fg)
            # Get our markers
            ret, markers = cv2.connectedComponents(sure_fg)
            # cv2.imshow("Found markers", markers)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1
            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0
            cv2.imshow("Markers", markers)
            markers = cv2.watershed(self.image,markers)
            print(markers)
            # cv2.imshow("Markers", markers)
            self.result = np.copy(self.image)
            self.result[markers == -1] = [255,0,0]
            cv2.imshow("Result", self.result)
            cv2.waitKey()

    def callback(self, tb_pos):
        self.threshold_value = tb_pos/self.alpha_slider_max*0.5
        print(self.threshold_value)
        cv2.imshow("Binary Threshold", self.gray_thresh)
        
        self.dist = cv2.distanceTransform(self.gray_thresh, cv2.DIST_L2, 5)
        # Normalize the distance image for range = {0.0, 1.0}
        # so we can visualize and threshold it
        cv2.normalize(self.dist, self.dist, 0, 1.0, cv2.NORM_MINMAX)
        cv2.imshow("Distance transform image", self.dist)

        _, self.dist = cv2.threshold(self.dist, self.threshold_value, 1.0, cv2.THRESH_BINARY)

        # Dilate a bit the dist image
        kernel1 = np.ones((3,3), dtype=np.uint8)
        self.dist = cv2.dilate(self.dist, kernel1)
        cv2.imshow('Peaks', self.dist)

        self.dist_8u = self.dist.astype('uint8')
        # Find total markers
        _, self.contours, _ = cv2.findContours(self.dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create the marker image for the watershed algorithm
        self.markers = np.zeros(self.dist.shape, dtype=np.int32)  

        # Draw the foreground markers
        for i in range(len(self.contours)):
            cv2.drawContours(self.markers, self.contours, i, (i+1), -1)

        # Draw the background marker
        cv2.circle(self.markers, (5,5), 3, (255,255,255), -1)
        cv2.imshow('Main Markers', self.markers*10000)

        cv2.watershed(self.imgResult, self.markers)
        self.mark = self.markers.astype('uint8')
        self.mark = cv2.bitwise_not(self.mark)
        cv2.imshow('Markers_v2', self.mark)
        # Generate random colors
        # self.colors = []
        # for contour in self.contours:
        #     self.colors.append((random.randint(0,256), random.randint(0,256), random.randint(0,256)))


        # Create the result image
        # self.dst = np.zeros((self.markers.shape[0], self.markers.shape[1], 3), dtype=np.uint8)
        # # Fill labeled objects with random colors
        # for i in range(self.markers.shape[0]):
        #     for j in range(self.markers.shape[1]):
        #         index = self.markers[i,j]
        #         if index > 0 and index <= len(self.contours):
        #             self.dst[i,j,:] = self.colors[index-1]
        # # Visualize the final image
        # cv2.imshow('Final Result', self.dst)

