# !/usr/local/bin/python3
# @Time : 2021/4/25 17:43
# @Author : Tianlei.Shi
# @Site :
# @File : image_process.py
# @Software : PyCharm
import cv2
import tensorflow as tf
import skimage
from skimage import transform

save_path_profix = "E:\\Depth-estimation\\images\\image_db\\"
save_path_num = 1
save_path_pastfix = ".jpg"

def test(srcpath, despath):
    '''
    for test new function
    :param srcpath: source file path
    :param despath: destination save path
    :return:
    '''
    # 读入图片
    img = tf.io.read_file(srcpath)

    # 解码为tensor格式
    img = tf.image.decode_jpeg(img)
    noise = tf.cast(tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=1.0,
                             dtype=tf.float32), dtype=tf.uint8)

    img = tf.cast(img, dtype=tf.uint8)

    img = tf.add(img, noise)

    img = tf.image.encode_jpeg(img)

    with tf.io.gfile.GFile(despath, 'wb') as file:
        file.write(img.numpy())



def rawPic(pic):
    '''
    save raw picture
    :param pic: picture
    :return:
    '''
    img = tf.cast(pic, dtype=tf.uint8)
    img = tf.image.encode_jpeg(img)

    global save_path_num
    num = str(save_path_num)
    path = save_path_profix + num + save_path_pastfix
    with tf.io.gfile.GFile(path, 'wb') as file:
        file.write(img.numpy())
    save_path_num += 1


def flip(pic):
    '''
    flip picture, img = tf.image.flip_left_right(pic)
    :param pic:
    :return:
    '''
    img = tf.image.flip_left_right(pic)
    img = tf.cast(img, dtype=tf.uint8)
    img = tf.image.encode_jpeg(img)

    global save_path_num
    num = str(save_path_num)
    path = save_path_profix + num + save_path_pastfix
    with tf.io.gfile.GFile(path, 'wb') as file:
        file.write(img.numpy())
    save_path_num += 1


def rotation(pic):
    '''
    rotate picture, img = tf.image.rot90(pic, k=2) to rotate 180
    :param pic:
    :return:
    '''
    img = tf.image.rot90(pic, k=2)
    img = tf.cast(img, dtype=tf.uint8)
    img = tf.image.encode_jpeg(img)

    global save_path_num
    num = str(save_path_num)
    path = save_path_profix + num + save_path_pastfix
    with tf.io.gfile.GFile(path, 'wb') as file:
        file.write(img.numpy())
    save_path_num += 1


def GaussianNoise(pic):
    '''
    add Gaussian Noise on pic, img = tf.add(img, noise)
    :param pic:
    :return:
    '''
    noise = tf.cast(tf.random.normal(shape=tf.shape(pic), mean=0.0, stddev=1.0,
                                     dtype=tf.float32), dtype=tf.uint8)
    img = tf.cast(pic, dtype=tf.uint8)

    img = tf.add(img, noise)

    img = tf.cast(img, dtype=tf.uint8)
    img = tf.image.encode_jpeg(img)

    global save_path_num
    num = str(save_path_num)
    path = save_path_profix + num + save_path_pastfix
    with tf.io.gfile.GFile(path, 'wb') as file:
        file.write(img.numpy())
    save_path_num += 1


file_path = r'E:\\Depth-estimation\\images\\images\\1\\'

# test("E:\\Depth-estimation\\images\\images\\0.jpg")

import os
fileList = os.listdir(file_path)

count = 0

for i in fileList:
    img_path = file_path + i

    # 读入图片
    img = tf.io.read_file(img_path)

    # 解码为tensor格式
    img = tf.image.decode_jpeg(img)

    rawPic(img)
    flip(img)
    rotation(img)
    GaussianNoise(img)

    count += 1

    if count % 30 == 0:
        print("now is:", count)

print("finished",count)

