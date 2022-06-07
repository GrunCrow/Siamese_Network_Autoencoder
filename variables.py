small = True

if small:
    folder_left_dir = "Images/left_small"
    folder_right_dir = "Images/right_small"

    HEIGHT = 49     # alto
    WIDTH = 40  # ancho
    NUM_COLORS = 3

    NUM_FEATURES = HEIGHT * WIDTH * NUM_COLORS

    DENSITY = (WIDTH * HEIGHT) / 10
else:
    folder_left_dir = "Images/left"
    folder_right_dir = "Images/right"

    HEIGHT = 245  # alto
    WIDTH = 200  # ancho
    NUM_COLORS = 3

    NUM_FEATURES = HEIGHT * WIDTH * NUM_COLORS

    DENSITY = (WIDTH * HEIGHT) / 10

left_images = []
left_images_train = []
left_images_test = []

right_images = []
right_images_train = []
right_images_test = []


