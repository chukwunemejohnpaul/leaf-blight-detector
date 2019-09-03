import cv2
import numpy as np
from skimage import segmentation, color
from skimage.io import imread
from skimage.future import graph
from skimage import data


image = imread('ricedisease.jpg')
blurred = cv2.GaussianBlur(image,(5,5),-1)

img_segments = segmentation.slic(blurred, compactness=20, n_segments=200)
superpixels = color.label2rgb(img_segments,image, kind='avg')


segment_graph = graph.rag_mean_color(image, img_segments, mode='similarity')
img_cuts = graph.cut_normalized(img_segments, segment_graph)
normalized_cut_segments = color.label2rgb(img_cuts, image, kind='avg')


cv2.imshow("pixels",normalized_cut_segments)
k = cv2.waitKey(0)
cv2.destroyAllWindows()