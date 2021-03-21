import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model_best = load_model('./best_model_class50.hdf5', compile = False)

food_list = []
with open('./classes.txt', 'r') as txt:
  paths = [read.strip() for read in txt.readlines()]
  for p in paths:
    food_list.append(p)
print(food_list)

def predict_class(model, images, show = True):
  for img in images:
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)
    food_list.sort()
    pred_value = food_list[index]
    pred_proba = pred[0][index]
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.suptitle(pred_value)
        plt.title(pred_proba)
        plt.show()

images = []
images.append('test1.jpg')
images.append('test2.jpg')
images.append('test3.jpg')

predict_class(model_best, images, True)


