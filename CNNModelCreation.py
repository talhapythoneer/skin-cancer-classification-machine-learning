from keras.preprocessing.image import ImageDataGenerator

imageGen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

trainDatasetPath = r"C:\Users\Zayn\Desktop\DIP Research Work\Datasets\final\train"
testDatasetPath = r"C:\Users\Zayn\Desktop\DIP Research Work\Datasets\final\test"

trainImgGen = imageGen.flow_from_directory(trainDatasetPath,
                                           target_size=(150, 150),
                                           batch_size=16,
                                           class_mode='binary')

testImgGen = imageGen.flow_from_directory(testDatasetPath,
                                          target_size=(150, 150),
                                          batch_size=16,
                                          class_mode='binary')

results = model.fit_generator(
    trainImgGen,
    epochs=120,
    steps_per_epoch=150,
    validation_data=testImgGen,
    validation_steps=12,
)

model.save("Final_E120.h5")
print(model.summary())
print(results.history)
