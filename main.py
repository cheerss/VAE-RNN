import tensorflow as tf
import vgg
import numpy as np
from utils import *
from sys import stderr
from operator import mul

DEBUG = False
width, height, channel = 128, 128, 3
img_size = width * height
enc_size = dec_size = 256
z_size = 10
batch_size = 1
T = 10
BUILT = False
atten = False
encoder = tf.contrib.rnn.core_rnn_cell.LSTMCell(enc_size, state_is_tuple=True)
decoder = tf.contrib.rnn.core_rnn_cell.LSTMCell(dec_size, state_is_tuple=True)
data_path= "vgg.mat"
learning_rate = 0.005
ratio = 1e4
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def main():
	g = tf.Graph()
	with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
		x = tf.placeholder(tf.float32, shape=(1, img_size * channel))
		# x_reconstr = tf.placeholder(tf.float32, shape=(1, img_size, channel))

		img = imread("img/1-style.jpg")
		stderr.write('img shape: ' + str(img.shape) + '\n')
		img_resize = imresize(img, [width, height])
		img_float = img_resize.astype('float') / 255
		imsave('img_resize/1-style.jpg', img_resize)
		stderr.write('shape of img_float: ' + str(img_float.shape) + '\n')

		x_reconstr, Lz = reconstruct(x)
		x_reshape = tf.reshape(x, [batch_size, width, height, channel])
		x_reconstr_reshape = tf.reshape(x_reconstr, [batch_size, width, height, channel])
		Ls = ratio * style_loss(x_reshape, x_reconstr_reshape)
		loss = Ls + Lz

		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		sess.run(tf.global_variables_initializer())

		feed_dict = { x: np.reshape(img_float, [1, -1]) }
		for i in range(1000):
			sess.run(train_step, feed_dict)
			stderr.write("Lz is " + str(Lz.eval(feed_dict)) + "  Ls is " + str(Ls.eval(feed_dict)) + "\n")
			if i % 10 == 0:
				img_reconstr = x_reconstr_reshape.eval(feed_dict)
				img_reconstr = (img_reconstr * 255).astype('int')
				imsave('img_reconstr/1-style-' + str(i) + '.jpg', np.reshape(img_reconstr, [width, height, channel]))


def style_loss(x, x_reconstr):
	style_features = {}
	net, mean = vgg.net(data_path, x)
	for layer in STYLE_LAYERS:
		feature = net[layer]
		size_origin = feature.get_shape().as_list()
		stderr.write('origin size is ' + str(size_origin) + '\n')
		feature = tf.reshape(feature, (-1, size_origin[3]))
		gram = tf.matmul(tf.transpose(feature), feature) / reduce(mul, size_origin, 1)
		style_features[layer] = gram

	x_grams = {}
	net, mean = vgg.net(data_path, x_reconstr)
	for layer in STYLE_LAYERS:
		feature = net[layer]
		size_origin = feature.get_shape().as_list()
		feature = tf.reshape(feature, (-1, size_origin[3]))
		gram = tf.matmul(tf.transpose(feature), feature) / reduce(mul, size_origin, 1)
		x_grams[layer] = gram

	Ls = 0
	for layer in STYLE_LAYERS:
		# stderr.write(str(reduce(mul, x_grams[layer].get_shape().as_list(), 1)))
		Ls += tf.nn.l2_loss(x_grams[layer] - style_features[layer]) / reduce(mul, x_grams[layer].get_shape().as_list(), 1)

	return Ls


def reconstruct(x):
	global BUILT
	c = [0] * T
	mus, sigmas, logsigmas = [0] * T, [0] * T, [0] * T
	h_dec_prev = tf.zeros((1, dec_size), dtype=tf.float32)
	enc_state = encoder.zero_state(1, tf.float32)
	dec_state = decoder.zero_state(1, tf.float32)

	for t in range(T):
		c_prev = tf.zeros((1, img_size * channel)) if t == 0 else c[t-1]
		x_hat = x - tf.sigmoid(c_prev)
		r = read(x, x_hat, h_dec_prev, atten)
		h_enc, enc_state = encode(tf.concat([r, h_dec_prev], 1), enc_state)
		z, mus[t], logsigmas[t], sigmas[t] = sampleZ(h_enc)
		stderr.write('size_of z is :' + str(z) + '\n')
		h_dec, dec_state = decode(z, dec_state)
		h_dec_prev = h_dec
		c[t] = c_prev + write(h_dec, atten)
		stderr.write('size of c[t]: ' + str(c[t]) + '\n')
		BUILT = True
	x_reconstr = tf.sigmoid(c[-1])

	Lz = 0
	for t in range(T):
		Lz += tf.square(mus[t]) + tf.square(sigmas[t]) + tf.square(logsigmas[t])
	Lz = 0.5 * tf.reduce_mean(Lz)

	return x_reconstr, Lz


def linear(x, output_dim):
	w = tf.get_variable("w", [x.get_shape()[1], output_dim])
	b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
	return tf.matmul(x, w) + b


def encode(input, state):
	with tf.variable_scope("encoder", reuse=BUILT):
		return encoder(input, state)


def decode(input, state):
	with tf.variable_scope("decoder", reuse=BUILT):
		return decoder(input, state)


def sampleZ(h_enc):
	with tf.variable_scope("mu", reuse=BUILT):
		mu = linear(h_enc, z_size)
	with tf.variable_scope("sigma", reuse=BUILT):
		logsigma = linear(h_enc, z_size)
		sigma = tf.exp(logsigma)
	return tf.random_normal([batch_size, z_size], mean=mu, stddev=sigma, dtype=tf.float32), mu, logsigma, sigma

def read(x, x_hat, h_dec_prev, atten):
	if(atten == False):
		return tf.concat([x, x_hat], 1)
	else:
		return

def write(h_dec, atten):
	if(atten == False):
		with tf.variable_scope("write", reuse=BUILT):
			return linear(h_dec, img_size * channel)
	else:
		return


# train the network
def train():
	with tf.Session() as sess:
		image = tf.placeholder(tf.float32, shape = [width, height])

# generate a new picture
def generate():
	return


if __name__ == '__main__':
	main()