import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt
import sys
from skimage.io import imread
from tqdm import tqdm

'''OpenCV Template Matching (to find waldo)
Template Matching is the idea of sliding a target image(template) over a source image (input). 
The template is compared to the input. A match is determined by the how much the neighbourhood pixels in the input matches with the template
'''
img_rgb = cv2.imread('whereis.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('waldo.jpg',0)

# saves the width and height of the template into 'w' and 'h'
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# setting the threshold in case of multi objects.
threshold = 0.36
# finding the values where it exceeds the threshold, I set the value low because in the puzzle how clustered and similar people are 
loc = np.where( res >= threshold) 
for pt in zip(*loc[::-1]):
    #draw rectangle on places where it exceeds threshold
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
    
cv2.imwrite('found_waldo2.png',img_rgb)

#Template Matching from scratch


def template_matching(src, temp, method = 'mse'):
    
    h = src.shape[0]
    w = src.shape[1]

    ht = temp.shape[0]
    wt = temp.shape[1]


    score = np.empty((h - ht, w - wt))

    if method == 'mse':
        for dy in tqdm(range(0, h - ht)):
            for dx in range(0, w - wt):
                image_patch = src[dy:dy + ht, dx:dx + wt]
                diff = np.mean((image_patch - temp) ** 2) #mse
                score[dy, dx] = diff
                pt = np.unravel_index(score.argmin(), score.shape)
               
    if method == 'corr':
        for dy in tqdm(range(0, h - ht)):
            for dx in range(0, w - wt):
        
                diff = np.corrcoef(src[dy:dy + ht, dx:dx + wt].flat, temp.flat)[0, 1]
                score[dy, dx] = diff
               # pt = np.unravel_index(score.argmax(), score.shape)
                pt =np.where(corr_result == corr_result.max())   #same as line 59     
                cv2.rectangle(img, (pt[1], pt[0]), (pt[1] + wt, pt[0] + ht), (0, 0, 200), 3)
                cv2.imwrite('output.png', img)
    
    
    return(pt[1], pt[0]) 

def main(method = 'corr'):

    img = cv2.imread('whereis.jpg')
    temp = cv2.imread("waldo.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

    h, w = temp.shape
    pt = template_matching(gray, temp, method = 'corr')

    cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
    cv2.imwrite('output.png', img)
    
main('corr')
