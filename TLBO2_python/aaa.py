from keras.preprocessing.image import ImageDataGenerator
#instantiate the ImageDataGenerator class
datagen = ImageDataGenerator(
        rotation_range=4)
        # height_shift_range=0.2,
        # width_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # fill_mode='nearest')
#loop over the data in batches and this automatically saves the images
i = 0
for batch in datagen.flow_from_directory('Chaotic_TLBO/real_and_fake_face', batch_size=1, target_size=(600, 600),
                          save_to_dir='E:/New folder', save_format='jpg'):
    i += 1
    if i > 1:
        break