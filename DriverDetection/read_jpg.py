# http://askubuntu.com/questions/507459/pil-install-in-ubuntu-14-04-1-lts

import numpy
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("example.jpg").convert("L")
print img.size
imgarr = numpy.asarray(img)

print imgarr.shape

#plt.imshow(imgarr, cmap='gray')
#plt.show()

img.save("new.jpg")

img = Image.open("new.jpg")
print img.size
