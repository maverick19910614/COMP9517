# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:05:46 2020

@author: 
"""

#......IMPORT .........
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import copy


def task1(img):
    e = 0.001
    t = random.randint(0, 255)
    # t = 30
    time = 0
    t_l = []
    time_l = []
    while True:
        t_l.append(t)
        time_l.append(time)
        bi = img >= t
        bi_inv = img < t
        fore_pix = np.sum(bi)
        back_pix = np.sum(bi_inv)
        if fore_pix == 0:
            break
        if back_pix == 0:
            continue
        w0 = fore_pix / img.size
        u0 = np.sum(img * bi) / fore_pix
        w1 = back_pix / img.size
        u1 = np.sum(img * bi_inv) / back_pix
        if abs(t - (u0 + u1) / 2) <= e:
            break
        else:
            t = (u0 + u1) / 2
        time = time + 1
    # plt.xlabel('x axis Iteration')
    # plt.ylabel('y axis Threshold')
    # plt.plot(time_l, t_l)
    # plt.show()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= t:
                img[i][j] = 0
            else:
                img[i][j] = 255
    # ret,thresh = cv2.threshold(img,t,255,cv2.THRESH_BINARY)
    return img, t, time_l, t_l
    
    
    
    

def task2(img):
    new_img = []
    for i in range(img.shape[0] + 4):
        new_img.append([])
        for j in range(img.shape[1] + 4):
            new_img[i].append(255)
    new_img = np.array(new_img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i + 2][j + 2] = img[i][j]
    # for i in range(3,10,2)
    # median filter
    c = copy.copy(new_img)
    for i in range(2, new_img.shape[0] - 2):
        for j in range(2, new_img.shape[1] - 2):
            a = sorted([c[i - 2][j - 2], c[i - 2][j - 1], c[i - 2][j], c[i - 2][j + 1], c[i - 2][j + 2],
                        c[i - 1][j - 2], c[i - 1][j - 1], c[i - 1][j], c[i - 1][j + 1], c[i - 1][j + 2],
                        c[i][j - 2], c[i][j - 1], c[i][j], c[i][j + 1], c[i][j + 2],
                        c[i + 1][j - 2], c[i + 1][j - 1], c[i + 1][j], c[i + 1][j + 1], c[i + 1][j + 2],
                        c[i + 2][j - 2], c[i + 2][j - 1], c[i + 2][j], c[i + 2][j + 1], c[i + 2][j + 2]])
            new_img[i][j] = a[12]

    # First pass
    img_c = copy.copy(new_img)
    t = 1
    d = {}
    d[t] = t
    for i in range(1, img_c.shape[0] - 1):
        for j in range(1, img_c.shape[1] - 1):
            #             if img_c[i][j] > 0:
            #                 img_c[i][j]=t
            #     for i in range(1,img_c.shape[0]-1):
            #         for j in range(1,img_c.shape[1]-1):
            if img_c[i][j] < 255:
                if img_c[i - 1][j - 1] < 255 or img_c[i - 1][j] < 255 or img_c[i - 1][j + 1] < 255 or img_c[i][j - 1] < 255:
                    a = [img_c[i - 1][j - 1], img_c[i - 1][j], img_c[i - 1][j + 1], img_c[i][j - 1]]
                    a = [k for k in a if k < 255]
                    img_c[i][j] = min(a)
                else:
                    img_c[i][j] = t
                    t = t + 1
                    d[t] = t
    for i in range(1, img_c.shape[0] - 1):
        for j in range(1, img_c.shape[1] - 1):
            if img_c[i][j] < 255:
                a = [img_c[i - 1][j - 1], img_c[i - 1][j], img_c[i - 1][j + 1],
                     img_c[i][j - 1], img_c[i][j], img_c[i][j + 1],
                     img_c[i + 1][j - 1], img_c[i + 1][j], img_c[i + 1][j + 1], d[img_c[i][j]]]
                d[img_c[i][j]] = min([k for k in a if k < 255])
    new_d = dict()
    for keys, values in d.items():
        new_d[values] = []
    for keys, values in d.items():
        new_d[values].append(keys)
    key = sorted(new_d.keys())
    key.reverse()
    for i in key:
        for j in new_d[i]:
            if j != i and j in new_d.keys():
                new_d[i].extend(new_d.pop(j))
    new_d.pop(sorted(new_d.keys())[-1])

    nums = len(new_d)

    return new_img, nums, new_d, img_c




def task3(new_d,img_c,min_area):
    dd = dict()
    l = len(new_d)
    for key,value in new_d.items():
        dd[key] = 0
        for i in value:
            dd[key] += np.sum(img_c==i)
    s = []
    ll = []
    for key,value in dd.items():
        if dd[key] >= min_area:
            s.append(key)
    for i in s:
        ll.extend(new_d[i])
    for i in range(img_c.shape[0]):
        for j in range(img_c.shape[1]):
            if img_c[i][j] in ll:
                img_c[i][j] = 0
            else:
                img_c[i][j] = 255
    re = round((l-len(s))/l,4)
    return re,img_c

    
    
    
    

    
my_parser = argparse.ArgumentParser()
my_parser.add_argument('-o','--OP_folder', type=str,help='Output folder name', default = 'OUTPUT')
my_parser.add_argument('-m','--min_area', type=int,action='store', required = True, help='Minimum pixel area to be occupied, to be considered a whole rice kernel')
my_parser.add_argument('-f','--input_filename', type=str,action='store', required = True, help='Filename of image ')
# Execute parse_args()
args = my_parser.parse_args()

output = args.OP_folder
m_area = args.min_area
input = args.input_filename

img = cv2.imread(input,0)
input = input.replace('.png','')
img,threshold,x,y = task1(img)
Threshold = 'Threshold' + ' = ' + str(round(threshold,2))
o0 = output+'/'+input+'_Task1(1).png'
o1 = output+'/'+input+'_Task1(2).png'
o11 = input+'_Task1(2).png'

#plt.subplot(121)
plt.figure(figsize = (12,8))
plt.xlabel('x axis Iteration')
plt.ylabel('y axis Threshold')
plt.plot(x, y)
plt.xticks(range(len(x)))
#for i in y:
plt.plot(x,y,marker='o')
for xy in zip(x,y):
    plt.annotate('%.2f'%xy[1],xy=xy,xytext=(-10, 10),textcoords='offset points ')
#plt.savefig(o0)
plt.savefig(o0)
plt.show()


#plt.subplot(122)
plt.imshow(img,cmap = 'gray')
plt.title(Threshold,fontsize=10)
plt.xlabel(o11)
plt.ylabel('')
plt.xticks([])
plt.yticks([])
plt.savefig(o1)



#cv2.imwrite(o,img)
#filtered_image,nums = task(img)
img,nums,d,img_c = task2(img)
num = 'Number of rice kernals' + ' = ' + str(nums)
o2 = output + '/' + input + '_Task2.png'
o21 = input + '_Task2.png'

plt.imshow(img,cmap = 'gray')
plt.title(num,fontsize=10)
plt.xlabel(o21)
plt.ylabel('')
plt.xticks([])
plt.yticks([])
plt.savefig(o2)

re,img = task3(d,img_c,m_area)
per = 'Percentage of damaged rice kernels' + ' = ' + str(re)
o3 = output + '/' + input + '_Task3.png'
o31 = input + '_Task3.png'

plt.imshow(img,cmap = 'gray')
plt.title(per,fontsize=10)
plt.xlabel(o31)
plt.ylabel('')
plt.xticks([])
plt.yticks([])
plt.savefig(o3)