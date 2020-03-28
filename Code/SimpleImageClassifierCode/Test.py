import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing import image

# load model
model = load_model('C:/GIT/CNN/Code/SimpleImageClassifierCode/model.h5')

#test_image = image.load_img('C:/GIT/CNN/Code/SimpleImageClassifierCode/Images/Images/Shiv/IMG_3011.jpg', target_size = (64, 64))
test_image = image.load_img('C:/GIT/CNN/Code/SimpleImageClassifierCode/Images/Images/Siddhant/IMG_3213.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)


if result[0][0] == 1:
    prediction = 'Siddhant'
    print(prediction)
else:
    prediction = 'Shiv'
    print(prediction)