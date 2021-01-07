#Realized by Alexandru-Andrei Carmici and Mihai Necula

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import path
import random

random.seed(0)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

name = input("Write image name or EXIT to exit: ")

if name == "EXIT":
    exit()
else:
    way="./images/"+name
    backupway="../images/"+name

    if not path.exists(way) and not path.exists(backupway):
        print("\nERROR! FILE DOES NOT EXISTS!")
        input()
        exit()
    elif path.exists(backupway):
        way = backupway

    Img =  mpimg.imread(way)
    plt.axis("off")
    plt.title(name)
    plt.imshow(Img)
    plt.show()

    grayImg = rgb2gray(Img)

    U,S,V = np.linalg.svd(grayImg)

    sigmas = np.diag(S)

    rank = random.randint(0,100)
    print("Rank: ",rank)
    aprox_img = U[:,:rank] @ np.diag(S[:rank]) @ V[:rank,:]
    plt.axis("off")
    plt.title("Compressed "+name)
    plt.imshow(aprox_img, cmap=plt.get_cmap('gray'), vmin = 0, vmax = 1)
    plt.show()