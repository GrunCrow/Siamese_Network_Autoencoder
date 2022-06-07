import os
import cv2
import splitfolders

from variables import left_images, right_images


#                                                              Load images:
def load_images_from_folders(left_folder, right_folder):
    for left_filename in os.listdir(left_folder):
        left_img = cv2.imread(os.path.join(left_folder, left_filename))
        if left_img is not None:
            left_images.append(left_img)
            # print(left_img)
    for right_filename in os.listdir(right_folder):
        right_img = cv2.imread(os.path.join(right_folder, right_filename))
        if right_img is not None:
            right_images.append(right_img)
            # print(right_img)


def load_images_from_folders_simultaneously(left_folder, right_folder):
    for left_filename, right_filename in zip(os.listdir(left_folder), os.listdir(right_folder)):
        left_img = cv2.imread(os.path.join(left_folder, left_filename))
        right_img = cv2.imread(os.path.join(right_folder, right_filename))
        if left_img is not None:
            left_images.append(left_img)
            # print(left_img)
        if right_img is not None:
            right_images.append(right_img)
            # print(right_img)


def load_images_from_folder(folder):
    for left_filename in os.listdir(folder):
        left_img = cv2.imread(os.path.join(folder, left_filename))
        if left_img is not None:
            # print(left_img)
            left_images.append(left_img)


#                                                  Split images intro train, test
def split_set(img_set):
    n_train = len(img_set) * 0.8    # 6016 * 0.2 = 1203
    n_test = len(img_set) * 0.2     # 6016 * 0.8 = 4813

    train_set = []
    test_set = []

    n_img = 0
    for img in img_set:
        if n_img < round(n_train):
            train_set.append(img)
        else:
            test_set.append(img)

        n_img += 1

    '''print(len(img_set))
    print(len(train_set))
    print(round(n_train))
    print(len(test_set))
    print(round(n_test))    # round down -> int, to round correctly use round()

    print(len(train_set) + len(test_set) == len(img_set))'''

    assert len(train_set) == round(n_train)
    assert len(train_set) + len(test_set) == len(img_set)
    assert len(test_set) == round(n_test)

    return train_set, test_set

