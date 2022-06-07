import matplotlib.pyplot as plt
import cv2
import numpy as np
import os  # to load images
import keras

from load_images import load_images_from_folders, split_set
from variables import folder_left_dir, folder_right_dir, left_images, right_images, \
    left_images_train, left_images_test, right_images_train, right_images_test, \
    WIDTH, HEIGHT, NUM_FEATURES, NUM_COLORS, DENSITY
from autoencoder import autoencoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # for tensorflow

# load_images_from_folders_simultaneous(folder_left_dir, folder_right_dir)

load_images_from_folders(folder_left_dir, folder_right_dir)

'''plt.imshow(left_images[0])
plt.show()'''

# To get the size of the images:
# print(left_images[0].shape)     # 49x40 = 1960 features and 3 channels (RGB)

# left_images_simplify = left_images/255.0     # to keep data between 0 and 1
'''left_images_simplify = np.divide(left_images, 255.0)
right_images_simplify = np.divide(right_images, 255.0)'''

# left_images_simplify = left_images/255.0

left_images_train, left_images_test = split_set(left_images)
right_images_train, right_images_test = split_set(right_images)

# we need the data set to be a np array and we  /255 to have values betw 0 and 1:
left_images_train = np.array(left_images_train) / 255.0
left_images_test = np.array(left_images_test) / 255.0
right_images_train = np.array(right_images_train) / 255.0
right_images_test = np.array(right_images_test) / 255.0

autoencoder, encoder = autoencoder()

# train
epochs = 3

for epoch in range(epochs):
    history = autoencoder.fit(
        left_images_train,
        right_images_train,
        epochs=1,
        batch_size=32,
        validation_split=0.10
    )
    autoencoder.save(f"models/AE-{epoch + 1}.model")

'''example = encoder.predict([left_images_test[0].reshape(-1, HEIGHT, WIDTH, NUM_COLORS)])[0]
print(example)

print(example.shape)   # = (196,) = 14^2

plt.imshow(example.reshape(14, 14))
plt.title("encoded example")
plt.show()

plt.imshow(left_images_test[0])
plt.title("what the example was")
plt.show()

ae_out = autoencoder.predict([left_images_test[0].reshape(-1, HEIGHT, WIDTH, NUM_COLORS)])[0]
plt.imshow(ae_out)
plt.title("decoded image")
plt.show()'''

# iteratively
for d in left_images_test[:5]:  # just show 5 examples

    ae_out = autoencoder.predict([d.reshape(-1, HEIGHT, WIDTH, NUM_COLORS)])
    img = ae_out[0]

    plt.imshow(img)
    plt.title("decoded")
    plt.show()
    plt.imshow(np.array(d))
    plt.title("original")
    plt.show()

print("end")
