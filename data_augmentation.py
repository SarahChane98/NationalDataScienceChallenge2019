from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
import random
from math import pi
import tensorflow as tf

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_samples = pd.read_csv('data/train.csv')


def rotate_images(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(128, 128, 3))
    radian = tf.placeholder(tf.float32, shape=(len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict={X: X_imgs, radian: radian_arr})
            X_rotate.append(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    print(X_rotate.shape)
    return X_rotate


def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(128, 128, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

    flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict={X: X_imgs})
    X_flip.append(flipped_imgs)
    X_flip = np.array(X_flip, dtype=np.float32)
    print(X_flip.shape)
    return X_flip


def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    gaussian = np.random.random((row, col, 1)).astype(np.float32)
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
    gaussian_img = cv2.addWeighted(X_imgs, 0.75, 0.25 * gaussian, 0.25, mean)
    gaussian_noise_imgs.append(gaussian_img)

    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    print(gaussian_noise_imgs.shape)
    return gaussian_noise_imgs


def create_training_data(dataset, root_dir, img_size, aug = False ):
    '''
    returns training data [image data, image label]
    '''

    training_data = []
    for i in range(len(dataset)):

        class_num = dataset['Category'][i]
        if dataset['image_path'][i][-4] != '.':
            img_path = root_dir + str(dataset['image_path'][i]) + ".jpg"
        else:
            img_path = root_dir + str(dataset['image_path'][i])
        img_array = cv2.imread(img_path)
        new_array = cv2.resize(img_array, (img_size, img_size))

        if aug:
            print("{}/{} - AUG".format(i, len(dataset)))
            rotated_imgs = rotate_images(new_array, -90, 90, 14)
            for j in range(len(rotated_imgs)):
                training_data.append([rotated_imgs[j], class_num])

            flipped_images = flip_images(new_array)
            for k in range(len(flipped_images)):
                training_data.append([flipped_images[k], class_num])

            gaussian_noise_imgs = add_gaussian_noise(new_array)
            for l in range(len(gaussian_noise_imgs )):
                training_data.append([gaussian_noise_imgs[l], class_num])

        else:
            print("{}/{} - NORM".format(i, len(dataset)))
            training_data.append([new_array, class_num])

    random.shuffle(training_data)
    X = []
    y = []

    for features, labels in training_data:
        X.append(features)
        y.append(labels)
    X = np.array(X).reshape(-1, img_size, img_size, 3)
    y = np.array(y)
    return X, y


def select_data(dataset):
    # to balance the train dataset
    # find the max count of categories
    # make the rest up to this max count
    # this function calculates the number of data that needs augmentation in each category
    # so that total number of pictures in each category = max count
    max_count = 100

    df_by_categories_aug= []
    df_no_aug = []
    for i in range(58):
        df = dataset.loc[dataset['Category'] == i]
        count = len(df)
        if count >= max_count:
            df = df.sample(frac=max_count/count).reset_index(drop=True)
            df_no_aug.append(df)
        else:
            df_no_aug.append(df)
            no_of_data_for_augmentation = (max_count - count) / 18
            df_aug = df.sample(frac=no_of_data_for_augmentation / count, replace = True).reset_index(drop=True)
            df_by_categories_aug.append(df_aug)

    df_aug= pd.concat(df_by_categories_aug, ignore_index=True)
    df_no_aug = pd.concat(df_by_categories_aug, ignore_index=True)

    return df_aug, df_no_aug


selected_aug_data, no_aug_data = select_data(train_samples)

# to find number of pictures in each category
# train_samples = train_samples.groupby('Category').nunique()
# print(train_samples)

aug_train_images, aug_train_labels = create_training_data(selected_aug_data, 'data/', 128, aug=True)
normal_train_images, normal_train_labels = create_training_data(no_aug_data, 'data/', 128, aug=False)

train_images = np.concatenate((aug_train_images,normal_train_images))
train_labels = np.concatenate((aug_train_labels,normal_train_labels))

np.save("processed_data/augmented_images.npy", train_images)
np.save("processed_data/augmented_labels.npy", train_labels)

print(len())


# test_images, test_itemid = create_training_data(test_samples, 'data/', 128)

# np.save("processed_data/test_image_2.npy", test_images, allow_pickle=True)
# np.save("processed_data/test_itemid_2.npy", test_itemid, allow_pickle=True)
