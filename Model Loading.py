from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
model = load_model("SkinHairRemoved_E120.h5")

path = "malignant/73.jpg"

targetImg = "Datasets/original/test/" + path
h_targetImg = "Datasets/hair_removed/test/" + path

targetImgC = cv2.imread(targetImg)
targetImgC = cv2.cvtColor(targetImgC, cv2.COLOR_BGR2RGB)

h_targetImgC = cv2.imread(h_targetImg)
h_targetImgC = cv2.cvtColor(h_targetImgC, cv2.COLOR_BGR2RGB)

img = image.load_img(h_targetImg, target_size=(150, 150))

img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255

res = model.predict_classes(img)

print(res)
if res[0][0] == 0:
    print("It is Benign!")
    msg = "It is Benign!"
else:
    print("It is Malignant!")
    msg = "It is Malignant!"

print(model.predict(img))

f, axarr = plt.subplots(1, 2)
plt.title(msg)
plt.title(msg)
axarr[0].imshow(targetImgC)
plt.title(msg)
axarr[1].imshow(h_targetImgC)
plt.show()