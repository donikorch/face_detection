from skimage import data
from skimage.feature import Cascade
from skimage import io

import matplotlib.pyplot as plt
from matplotlib import patches

trained_file = data.lbp_frontal_face_cascade_filename()
detector = Cascade(trained_file)
image = io.imread('C:/Users/danya/Downloads/images/image_11.jpg')

detected = detector.detect_multi_scale(img=image,
                                       scale_factor=1.2,
                                       step_ratio=1,
                                       min_size=(200, 200),
                                       max_size=(2000, 2000))

plt.imshow(image)
img_desc = plt.gca()
plt.set_cmap('gray')

for patch in detected:
    img_desc.add_patch(patches.Rectangle((patch['c'], 
                                          patch['r']),
                                         patch['width'],
                                         patch['height'],
                                         fill=False,
                                         color='r',
                                         linewidth=2))

plt.show()