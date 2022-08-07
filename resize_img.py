import glob
from operator import ne
from PIL import Image, ImageOps
import numpy as np
import imageio
import matplotlib.pyplot as plt



pic = imageio.imread('a.png')

print('Type of the image : ' , type(pic))
print()
print('Shape of the image : {}'.format(pic.shape))
print('Image Hight {}'.format(pic.shape[0]))
print('Image Width {}'.format(pic.shape[1]))
print('Dimension of Image {}'.format(pic.ndim))

print (pic)

imageio.imsave("pic.png",pic)

"""
new_img = np.array(img_array)
new_img[new_img<10] = new_img[np.where(new_img<10)] + 60

imageio.imsave("new_img.png",new_img)
   

#print (img_array)


    #image1 = img_array.save(file)

"""

