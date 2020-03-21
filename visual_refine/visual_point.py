import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mping
image = cv2.imread('/home/ouc/TXH/DenseFusion/visual_refine/ori.jpg')

cam_cx = 325.26110
cam_cy = 242.04899
cam_fx = 572.41140
cam_fy = 573.57043
K = np.array([[cam_fx, 0.0, cam_cx],
    [0.0, cam_fy, cam_cy],
     [0.0, 0.0, 1.0]], dtype=np.float)


def show_mask(image):
    d2d = np.loadtxt('/home/ouc/TXH/DenseFusion/visual/2d.txt').astype(np.int)
    w, h = d2d.shape
    for i in range(w):
        image = cv2.circle(image, center=(d2d[i][1], d2d[i][0]), radius=1, color=(255, 0, 0))
    cv2.imshow('mask', image)
    cv2.imwrite('/home/ouc/TXH/DenseFusion/visual/mask.jpg', image)
    cv2.waitKey()

import random
def show_3D(image, K, target, name):
    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    d2_list = []
    for i in range(target.shape[0]):
        d2 = np.transpose(np.matmul(K, np.transpose(target[i])))
        d2 = d2 / d2[2]
        d2_list.append(d2[:2])
    d3 = np.array(d2_list, dtype=np.int)
    for i in range(d3.shape[0]):
        image = cv2.circle(image, center=(d3[i][0], d3[i][1]), radius=1, color=color)

    cv2.imshow('{}'.format(name), image)
    cv2.imwrite('/home/ouc/TXH/DenseFusion/visual_refine/{}.jpg'.format(name), image)
    cv2.waitKey(1500)



name = 'target'
pred = np.loadtxt('/home/ouc/TXH/DenseFusion/visual_refine/{}.txt'.format(name))
show_3D(image, K, target=pred,name=name)
