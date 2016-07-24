import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#all_images = np.zeros((5, 1080, 1920))

basewidth = 300
img = Image.open("1.jpg")
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))

#all_images = np.zeros((5, 1080, 1920))
all_images = np.zeros((5, hsize, basewidth))

for i in range(1, 6):
  img = Image.open(str(i)+".jpg")
  img = img.resize((basewidth, hsize), Image.ANTIALIAS)
  imgarr = np.asarray(img) + i * 10
  all_images[i-1::] = imgarr

for i in range(1, 6):
  tmp_arr = all_images[i-1,:,:]
  print tmp_arr.shape
  plt.imshow(tmp_arr, cmap='gray')
  plt.show()
