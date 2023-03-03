import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

#longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

x = load_img("name_imagen_to_predict.jpeg", target_size=(64,64,3))
x = img_to_array(x)
x = np.expand_dims(x, axis=0)
prediccion = cnn.predict(x)
#training_dataset.class_indices

if prediccion[0][0]==1:
    print("neumonia")
else:
    print("normal")    


