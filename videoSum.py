import cv2
import sys
import os
import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import namedtuple
import time

fileName = "../datasample1.mov"

motionThreshold = 10

ColorMoments = namedtuple('ColorMoments', ['mean', 'stdDeviation', 'skewness'], verbose=False)

Shot = namedtuple('Shot', ['shotNumber', 'startingFrame', 'endingFrame', 'keyFrames', 'avgEntropyDiff', 'avgMotion'])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def getEntropy(histogram, totalPixels):
    entropy = 0
    for pixels in histogram:
        if pixels != 0:
            prob = float (pixels / totalPixels)
            entropy -= prob * math.log(prob, 2)
            # print prob
    return entropy

def getColorMoments(histogram, totalPixels):
    sum = 0
    for pixels in histogram:
        sum += pixels
    mean = float (sum / totalPixels)
    sumOfSquares = 0
    sumOfCubes = 0
    for pixels in histogram:
        sumOfSquares += math.pow(pixels-mean, 2)
        sumOfCubes += math.pow(pixels-mean, 3)
    variance = float (sumOfSquares / totalPixels)
    stdDeviation = math.sqrt(variance)
    avgSumOfCubes = float (sumOfCubes / totalPixels)
    skewness = float (avgSumOfCubes**(1./3.))
    return ColorMoments(mean, stdDeviation, skewness)

def getHistogramDiff(currHistogram, prevHistogram):
    diff = 0
    for i in range(len(currHistogram)):
        diff += math.pow(currHistogram[i]-prevHistogram[i], 2)
    if diff==0:
        return 1
    else:
        return diff 

def getHistogramRatio(currHistogramDiff, prevHistogramDiff):
    ratio = float (currHistogramDiff / prevHistogramDiff)
    if ratio<1:
        ratio = 1/ratio
    return ratio

def getEuclideanDistance(currColorMoments, prevColorMoments):
    distance = math.pow(currColorMoments.mean - prevColorMoments.mean, 2) + math.pow(currColorMoments.stdDeviation - prevColorMoments.stdDeviation, 2) + math.pow(currColorMoments.skewness - prevColorMoments.skewness, 2)
    return distance

def getMotion(currImage, prevImage):
    motion = 0
    for i in range(len(currImage)):
        for j in range(len(currImage[i])):
            if int (currImage[i][j]) - int (prevImage[i][j]) > motionThreshold:
                motion += 1
    motion = float (motion / (i+1)*(j+1))
    return motion

def sortShots(shots):
    maxKeyFrames = -1
    maxAvgEntropyDiff = -1
    maxAvgMotion = -1
    for shot in shots:
        if shot.keyFrames > maxKeyFrames :
            maxKeyFrames = shot.keyFrames
        if shot.avgEntropyDiff > maxAvgEntropyDiff :
            maxAvgEntropyDif = shot.avgEntropyDiff
        if shot.avgMotion > maxAvgMotion :
            maxAvgMotion = shot.avgMotion

    weights = []
    for shot in shots:
        weight = shot.keyFrames / float(maxKeyFrames) + shot.avgEntropyDiff / float(maxAvgEntropyDiff) + shot.avgMotion / float(maxAvgMotion)
        weights.append((shot.shotNumber, weight))

    print weights
    weights = sorted(weights, key=lambda x: x[1])
    print weights

def main():
    videoCap = cv2.VideoCapture(fileName)
    
    entropy = []
    histogramDiff = []
    histogramRatio = []
    entropyDiff = []
    euclideanDistance = []
    motion = []

    t0 = time.clock()

    i = 0
    success, image = videoCap.read()
    while success:
        height = len(image)
        width = len(image[0])
        # print width, height
        totalPixels = width * height
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # plt.imshow(grayImage, cmap = plt.get_cmap('gray'))
        # plt.show()
        # print grayImage.shape
        # print grayImage
        histogram = cv2.calcHist([grayImage],[0],None,[256],[0,256])
        # print len(histogram)
        entropy.append( getEntropy(histogram, totalPixels) )
        # print entropy[i]
        colorMoments = getColorMoments(histogram, totalPixels)
        # print colorMoments
        if i==0:
            histogramDiff.append(4000000)
            histogramRatio.append(200)
            entropyDiff.append(0)
            euclideanDistance.append(0)
            motion.append(0)
        else:
            histogramDiff.append( getHistogramDiff(histogram, prevHistogram) )
            histogramRatio.append( getHistogramRatio(histogramDiff[i], histogramDiff[i-1]) )
            entropyDiff.append( abs(entropy[i] - entropy[i-1]))
            euclideanDistance.append( getEuclideanDistance(colorMoments, prevColorMoments) )
            motion.append( getMotion(grayImage, prevGrayImage) )

        prevHistogram = histogram
        prevGrayImage = grayImage
        prevColorMoments = colorMoments

        i += 1
        success, image = videoCap.read()
        print i
        if i==40:            
            break

    meanEntropyDiff = sum(entropyDiff) / float(len(entropyDiff))
    meanHistogramRatio = sum(histogramRatio) / float(len(histogramRatio))
    meanEuclideanDistance = sum(euclideanDistance) / float(len(euclideanDistance))

    thresholdEntropyDiff = meanEntropyDiff
    thresholdHistogramRatio = meanHistogramRatio
    thresholdEuclideanDistance = meanEuclideanDistance

    totalFrames = i
    
    motionSum = 0
    entropyDiffSum = 0
    
    shots = []
    shotNumber = 0
    prevFrame = 0
    keyFrames = 0
    for i in range(totalFrames):
        if euclideanDistance[i] > thresholdEuclideanDistance:
            keyFrames += 1

        entropyDiffSum += entropyDiff[i]
        motionSum += motion[i]
        
        if entropyDiff[i] > thresholdEntropyDiff and histogramRatio[i] > thresholdHistogramRatio:
            if i<= prevFrame+25 and shotNumber!=0:
                currShot = shots[shotNumber-1]
                numberOfFrames = currShot.endingFrame - currShot.startingFrame + 1
                newAvgEntropyDiff = ((currShot.avgEntropyDiff * numberOfFrames) + entropyDiffSum) / (i - currShot.startingFrame)
                newAvgMotion = ((currShot.avgMotion * numberOfFrames) + motionSum) / (i - currShot.startingFrame)
                shots[shotNumber-1] = Shot(currShot.shotNumber, currShot.startingFrame, i-1, currShot.keyFrames + keyFrames, newAvgEntropyDiff, newAvgMotion)
            else:
                avgEntropyDiff = entropyDiffSum / float(i - prevFrame)
                avgMotion = motionSum / float(i - prevFrame)
                shots.append(Shot(shotNumber, prevFrame, i-1, keyFrames, avgEntropyDiff, avgMotion))
                shotNumber += 1
            
            keyFrames = 0
            motionSum = 0
            entropyDiffSum = 0
            prevFrame = i

    # Adding the last shot
    avgEntropyDiff = entropyDiffSum / float(i - prevFrame)
    avgMotion = motionSum / float(i - prevFrame)
    shots.append(Shot(shotNumber, prevFrame, i-1, keyFrames, avgEntropyDiff, avgMotion))
    shotNumber += 1

    orderedShots = sortShots(shots)
    
    print 'Time taken to run =', time.clock() - t0, 'seconds' 

    print 'Shots -' , shots
    print 'Entropy -', entropy, '\n'
    print 'HistogramDiff -', histogramDiff, '\n'
    print 'HistogramRatio -', histogramRatio, '\n'
    print 'EntropyDiff -', entropyDiff, '\n'
    print 'EuclideanDistance -', euclideanDistance, '\n'
    print 'Motion -', motion, '\n'
    print 'MeanEntropyDiff -', meanEntropyDiff
    print 'MeanHistogramRatio -', meanHistogramRatio
    print 'MeanEuclideanDistance -', meanEuclideanDistance

if __name__ == '__main__':
	    main()