#!/usr/bin/python

import sys, os, time
import math, cv2
import numpy as np



class InpainterV2():

    fillFront = []
    normals = []
    patchHeight = patchWidth = 0


    def __init__(self, inputImage, mask, halfPatchWidth = 16):
        self.inputImage = np.copy(inputImage)
        self.mask = np.copy(mask)
        self.workingMask = np.copy(mask)
        self.workingImage = np.copy(inputImage)
        self.result = np.ndarray(shape = inputImage.shape, dtype = inputImage.dtype)
        self.halfPatchWidth = halfPatchWidth

    #make sure dimensions and types match up
    def checkValidInputs(self):
        if self.inputImage.dtype != np.uint8 or self.mask.dtype != np.uint8 or self.mask.shape != self.inputImage.shape[:2] or self.halfPatchWidth == 0:
            return False
        return True


    def initMatrices(self):
        _, self.confidence = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        _, self.confidence = cv2.threshold(self.confidence, 2, 1, cv2.THRESH_BINARY_INV)
        
        self.sourceRegion = np.uint8(np.copy(self.confidence))
        #TODO remove
        #self.sourceRegion = np.uint8(self.sourceRegion)
        self.originalSourceRegion = np.copy(self.sourceRegion)
        
        self.confidence = np.float32(self.confidence)
 
        _, self.targetRegion = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        _, self.targetRegion = cv2.threshold(self.targetRegion, 2, 1, cv2.THRESH_BINARY)
        self.targetRegion = np.uint8(self.targetRegion)
        self.data = np.ndarray(shape = self.inputImage.shape[:2],  dtype = np.float32)
        
        self.lKernel = np.ones((3, 3), dtype = np.float32)
        self.lKernel[1, 1] = -8.0
        self.kernelX = np.zeros((3, 3), dtype = np.float32)
        self.kernelX[1, 0] = -1.0
        self.kernelX[1, 2] = 1.0
        self.kernelY = cv2.transpose(self.kernelX)


    def createGradients(self):
        srcGray = cv2.cvtColor(self.workingImage, cv2.COLOR_RGB2GRAY)
        #defualt params, maybe try something else?
        self.gradientX = cv2.Scharr(srcGray, cv2.CV_32F, 1, 0)
        self.gradientX = np.float32(cv2.convertScaleAbs(self.gradientX))
        self.gradientY = cv2.Scharr(srcGray, cv2.CV_32F, 0, 1)
        self.gradientY = np.float32(cv2.convertScaleAbs(self.gradientY))
    
        height, width = self.sourceRegion.shape
        for y in range(height):
            for x in range(width):
                if self.sourceRegion[y, x] == 0:
                    self.gradientX[y, x] = 0
                    self.gradientY[y, x] = 0
        
        self.gradientX /= 255
        self.gradientY /= 255

    def computeFillFront(self):
        # elements of boundryMat, whose value > 0 are neighbour pixels of target region. 
        boundryMat = cv2.filter2D(self.targetRegion, cv2.CV_32F, self.lKernel)

        #updated only what changes (mike said way easy)
        #TODO fix :) 
        sourceGradientX = cv2.filter2D(self.sourceRegion, cv2.CV_32F, self.kernelX)
        sourceGradientY = cv2.filter2D(self.sourceRegion, cv2.CV_32F, self.kernelY)
        
        del self.fillFront[:]
        del self.normals[:]
        height, width = boundryMat.shape[:2]
        for y in range(height):
            for x in range(width):
                if boundryMat[y, x] > 0:
                    self.fillFront.append((x, y))
                    dx = sourceGradientX[y, x]
                    dy = sourceGradientY[y, x]
                    
                    normalX, normalY = dy, - dx 
                    tempF = math.sqrt(pow(normalX, 2) + pow(normalY, 2))
                    if not tempF == 0:
                        normalX /= tempF
                        normalY /= tempF
                    self.normals.append((normalX, normalY))

    def computeConfidence(self):
        confidenceCopy = np.copy(self.confidence)
        for p in self.fillFront:
            pX, pY = p
            (aX, aY), (bX, bY) = self.getPatch(p)
            temp = np.where(self.targetRegion[aY:bY+1,aX:bX+1] == 0)
            total = np.sum(confidenceCopy[temp])
            self.confidence[pY, pX] = total / ((bX-aX+1) * (bY-aY+1))

    #return lower left and upper right coord of a patch
    def getPatch(self, point):
        centerX, centerY = point
        height, width = self.workingImage.shape[:2]
        minX = max(centerX - self.halfPatchWidth, 0)
        maxX = min(centerX + self.halfPatchWidth, width - 1)
        minY = max(centerY - self.halfPatchWidth, 0)
        maxY = min(centerY + self.halfPatchWidth, height - 1)
        upperLeft = (minX, minY)
        lowerRight = (maxX, maxY)
        return upperLeft, lowerRight

    def computeData(self):
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            currentNormalX, currentNormalY = self.normals[i]
            self.data[y, x] = math.fabs(self.gradientX[y, x] * currentNormalX + self.gradientY[y, x] * currentNormalY) + 0.001

    def computeTarget(self):
        self.targetIndex = 0
        maxPriority, priority = 0, 0
        #we care about data and confidence
        #params to tweak
        omega, alpha, beta = 0.7, 0.2, 0.8
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            Rcp = (1-omega) * self.confidence[y, x] + omega
            priority = alpha * Rcp + beta * self.data[y, x]
            
            if priority > maxPriority:
                maxPriority = priority
                self.targetIndex = i

    def computeBestPatch(self):
        #find best patch to use to fill our target patch
        minError = bestPatchVariance = math.inf
        currentPoint = self.fillFront[self.targetIndex]
        (aX, aY), (bX, bY) = self.getPatch(currentPoint)
        pHeight, pWidth = bY - aY + 1, bX - aX + 1
        height, width = self.workingImage.shape[:2]
        workingImage = np.array(self.workingImage, dtype = np.float32)
        
        if pHeight != self.patchHeight or pWidth != self.patchWidth:
            print ('patch size changed.')
            self.patchHeight, self.patchWidth = pHeight, pWidth
            area = pHeight * pWidth
            SUM_KERNEL = np.ones((pHeight, pWidth), dtype = np.uint8)
            convolvedMat = cv2.filter2D(self.originalSourceRegion, cv2.CV_8U, SUM_KERNEL, anchor = (0, 0))
            self.sourcePatchULList = []
            
            # sourcePatchULList: list whose elements is possible to be the UpperLeft of an patch to reference.
            tmp = np.where(convolvedMat[0:height - pHeight, 0:width - pWidth] == area)
            tmp = list(zip(tmp[0],tmp[1]))
            self.sourcePatchULList.extend(tmp)
            
        countedNum = 0.0
        self.targetPatchSList = []
        self.targetPatchTList = []
        
        # targetPatchSList & targetPatchTList: list whose elements are the coordinates of  origin/toInpaint pixels.
        for i in range(pHeight):
            for j in range(pWidth):
                if self.sourceRegion[aY+i, aX+j] == 1:
                    countedNum += 1.0
                    self.targetPatchSList.append((i, j))
                else:
                    self.targetPatchTList.append((i, j))
                
        for index, (y, x) in enumerate (self.sourcePatchULList):
                if (index % 1 != 0):
                    continue
                patchError = 0
                meanR = meanG = meanB = 0
                skipPatch = False


                tmpPatchSList = np.array(self.targetPatchSList)
                sourcePatchSList = tmpPatchSList + [y,x]
                targetPatchSList = tmpPatchSList + [aY,aX]

                sourceY= sourcePatchSList[:,0]
                sourceX= sourcePatchSList[:,1]

                targetY= targetPatchSList[:,0]
                targetX= targetPatchSList[:,1]

                meanR = np.mean(workingImage[sourceY,sourceX,0])
                meanG = np.mean(workingImage[sourceY,sourceX,1])
                meanB = np.mean(workingImage[sourceY,sourceX,2])

                patchError = np.sum(np.square(workingImage[sourceY,sourceX,:] - workingImage[targetY,targetX,:]))/countedNum

                alpha, beta = 0.9, 0.5
                if alpha * patchError <= minError:
                    patchVariance = 0
                    for (i, j) in self.targetPatchTList:
                                sourcePixel = workingImage[y+i][x+j]
                                difference = sourcePixel[0] - meanR
                                patchVariance += math.pow(difference, 2)
                                difference = sourcePixel[1] - meanG
                                patchVariance += math.pow(difference, 2)
                                difference = sourcePixel[2] - meanB
                                patchVariance += math.pow(difference, 2)
                    
                    # Use alpha & Beta to encourage path with less patch variance.
                    # For situations in which you need little variance.
                    # Alpha = Beta = 1 to disable.
                    if patchError < alpha * minError or patchVariance < beta * bestPatchVariance:
                        bestPatchVariance = patchVariance
                        minError = patchError
                        self.bestMatchUpperLeft = (x, y)
                        self.bestMatchLowerRight = (x+pWidth-1, y+pHeight-1)

    def updateMats(self):
        targetPoint = self.fillFront[self.targetIndex]
        tX, tY = targetPoint
        (aX, aY), (bX, bY) = self.getPatch(targetPoint)
        bulX, bulY = self.bestMatchUpperLeft
        pHeight, pWidth = bY-aY+1, bX-aX+1
        
        for (i, j) in self.targetPatchTList:
                    self.workingImage[aY+i, aX+j] = self.workingImage[bulY+i, bulX+j]
                    self.gradientX[aY+i, aX+j] = self.gradientX[bulY+i, bulX+j]
                    self.gradientY[aY+i, aX+j] = self.gradientY[bulY+i, bulX+j]
                    self.confidence[aY+i, aX+j] = self.confidence[tY, tX]
                    self.sourceRegion[aY+i, aX+j] = 1
                    self.targetRegion[aY+i, aX+j] = 0
                    self.workingMask[aY+i, aX+j] = 0
    
    def checkEnd(self):
        print(np.any(self.sourceRegion))
        
        height, width = self.sourceRegion.shape[:2]
        for y in range(height):
            for x in range(width):
                if self.sourceRegion[y, x] == 0:
                    return True
        return False


    def doInpaint(self):
        #one time inits
        self.initMatrices()
        self.createGradients()


        stay = True
        counter = 0
        while stay:
            counter += 1
            self.computeFillFront()
            self.computeConfidence()
            self.computeData()
            self.computeTarget()
            print ('start', time.asctime())
            self.computeBestPatch()
            print ('end :)', time.asctime())
            self.updateMats()
            stay = self.checkEnd()
            if counter % 20:
                cv2.imwrite("workingMask.jpg", self.workingMask)
                cv2.imwrite("workingImage.jpg", self.workingImage)
        
        self.result = np.copy(self.workingImage)
        cv2.imshow("Confidence", self.confidence)
