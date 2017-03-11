import tensorflow as tf
import scipy.misc
from sys import stderr
import numpy as np
import random

shape = [128, 128, 3]
data_path = 'vgg.mat'
eps = 1e-8
num_of_examples = 100
window_shape = [21, 21]
learning_rate = 0.01
iterations = 500
path = 'img_gen/'
mess_path = 'img_mess/'

def main():
    x = tf.placeholder(tf.float32, shape, name="x")
    img = open_and_resize('img/1-style.jpg', shape[0:2])
    # mean = np.mean(img)
    # x_texture = tf.sigmoid(tf.get_variable("texture", shape))

    losses = []
    for i in range(num_of_examples):
        content = mess(img, window_shape)
        content = np.reshape(content, [1, -1])
        x_texture = tf.reshape(x_texture, [1, -1])
        temp = tf.reduce_sum(binary_crossentropy(content, x_texture), 1)
        losses.append(temp)

    loss = reduce(tf.add, losses, 0.0)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            sess.run(train_step)
            stderr.write('loss: ' + str(sess.run(loss)) + '\n')
            if(i % 10 == 0):
                img_gen = (x_texture.eval() * 255).astype('int')
                img_gen = np.reshape(img_gen, shape)
                save_image(path + 'gen_' + str(i) + '.jpg', img_gen)
        img_gen = (x_texture.eval() * 255).astype('int')
        img_gen = np.reshape(img_gen, shape)
        save_image(path + 'gen_' + str(i) + '.jpg', img_gen)

    return

def open_and_resize(name, shape):
    img = scipy.misc.imread(name)
    img = scipy.misc.imresize(img, shape)
    return img

def save_image(path, img):
    scipy.misc.imsave(path, img)

def mess(img, window_shape):
    # img: [width, height, channel]
    ans = np.array([[[0] * 3] * (shape[1] + window_shape[1])] * (shape[0] + window_shape[0]))
    i = 0
    j = 0
    while(True):
        random_x = int((random.random() * shape[1]))
        random_y = int((random.random() * shape[0]))
        if (random_x - window_shape[1] / 2 < 0):
            random_x = window_shape[1]
        if (random_x + window_shape[1] / 2 >= shape[1]):
            random_x = shape[1] - window_shape[1] / 2 - 1
        if (random_y - window_shape[0] / 2 < 0):
            random_y = window_shape[0]
        if (random_y + window_shape[0] / 2 >= shape[0]):
            random_y = shape[0] - window_shape[0] / 2 - 1

        slice = img[random_y-window_shape[0]/2 : random_y+window_shape[0]/2 + 1, random_x-window_shape[1]/2 : random_x+window_shape[1]/2 + 1, :]
        ans[i: i+window_shape[1], j: j+window_shape[0], :] = slice
        i += window_shape[1]
        if(i >= shape[1]):
            i = 0
            j += window_shape[0]
        if(j >= shape[0]):
            break
    img_mess = ans[0: shape[0], 0: shape[1], 0: 3]
    assert img_mess.shape == (128,128,3)
    save_image(mess_path + 'gen_' + str(random.random()) + '.jpg', img_mess)
    return img_mess.astype('float') / 255


def binary_crossentropy(t, o):
    a = t * tf.log(o + eps)
    b = (1.0 - t) * tf.log(1.0 - o + eps)
    return -(a+b)


if __name__ == '__main__':
    main()