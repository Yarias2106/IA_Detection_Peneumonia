# Importar las liobrerías y paquetes
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import sys
import os
# Inicializar la CNN
classifier = Sequential()

# Paso 1 - Convolución
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), activation = "relu"))

# Paso 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Una segunda capa de convolución y max pooling
classifier.add(Conv2D(filters = 64,kernel_size = (3, 3), activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2,2)))

# Paso 3 - Flattening
classifier.add(Flatten())

# Paso 4 - Full Connection
classifier.add(Dense(units = 256, activation = "relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 2, activation = "softmax"))

# Compilar la CNN
classifier.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# Parte 2 - Ajustar la CNN a las imágenes para entrenar 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('./chest_xray/train',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='categorical')

testing_dataset = test_datagen.flow_from_directory('./chest_xray/test',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='categorical')

classifier.fit_generator(training_dataset,
                        steps_per_epoch=125,
                        epochs=15,
                        validation_data=testing_dataset,
                        validation_steps=50)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
classifier.save('./modelo/modelo.h5')
classifier.save_weights('./modelo/pesos.h5')



