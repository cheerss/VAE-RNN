import tensorflow as tf
import vgg
import numpy as np
from utils import *
from sys import stderr
from operator import mul

DEBUG = False
width, height, channel = 128, 128, 3
width_, height_, channel_ = 128, 128, 3
img_size = width_ * height_
enc_size = dec_size = 256
z_size = 60
batch_size = 1
T = 10
read_n = 30 # read glimpse grid width/height
write_n = 30 # write glimpse grid width/height
BUILT = False
atten = False
read_size = 2*read_n*read_n if atten else 2*img_size
write_size = write_n*write_n if atten else img_size
encoder = tf.contrib.rnn.core_rnn_cell.LSTMCell(enc_size, state_is_tuple=True)
decoder = tf.contrib.rnn.core_rnn_cell.LSTMCell(dec_size, state_is_tuple=True)
data_path= "../neural-style-bias/vgg.mat"
learning_rate = 0.0003
ratio_style = 1e-8
ratio_content = 1e0
eps=1e-8 # epsilon for numerical stability
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYERS = ('relu4_2', 'relu5_2')
save_path = "1-style-big/"
omiga = (1., 1., 1., 1., 1.)

def main():
	g = tf.Graph()
	with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
		global batch_size, width, height, channel, width_, height_, channel_
		# img_content = imresize(imread("img/1-content.jpg"), [width, height]).astype('float') / 255
		img_content = imread("img/1-content.jpg").astype('float') / 255
		# stderr.write('img shape: ' + str(img_content.shape) + '\n')
		# img_style = imresize(imread("img/1-style.jpg"), [width, height]).astype('float') / 255
		img_style = imread("img/1-style.jpg").astype('float') / 255
		# img_style = imread("img/3-style.jpg").astype('float') / 255
		imsave("img_resize/1-content.jpg", img_content)
		# stderr.write('shape of img_float: ' + str(img_float.shape) + '\n')
		width, height, channel = img_content.shape
		width_, height_, channel_ = img_style.shape

		x = tf.placeholder(tf.float32, shape=(batch_size, width_, height_, channel_))
		x_content = tf.placeholder(tf.float32, shape=(batch_size, width, height, channel))

		x_reconstr, Lz = reconstruct(x)
		Ls = ratio_style * style_loss(x_reconstr * 255, x * 255)
		# Lx = ratio_content * content_loss_(x_content, x_reconstr)
		loss = Ls + Lz #+ Lx

		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		sess.run(tf.global_variables_initializer())

		feed_dict = { x: np.reshape(img_style, [batch_size, width_, height_, channel_]), x_content: np.reshape(img_content, [batch_size, width, height, channel]) }
		for i in range(1000):
			sess.run(train_step, feed_dict)
			stderr.write("Lz: " + str(Lz.eval(feed_dict)))
			stderr.write(" Ls: " + str(Ls.eval(feed_dict)) + "\n")
			# stderr.write(" Lx is " + str(Lx.eval(feed_dict)) + "\n")
			if i % 10 == 0:
				img_reconstr = x_reconstr.eval(feed_dict)
				img_reconstr = (img_reconstr * 255).astype('int')
				imsave(save_path + str(i) + '.jpg', np.reshape(img_reconstr, [width_, height_, channel_]))

def content_loss_(x, x_reconstr):
	x_features = {}
	content_losses = []
	size = reduce(mul, x.get_shape().as_list())

	net, mean = vgg.net(data_path, x)
	for layer in CONTENT_LAYERS:
		x_features[layer] = net[layer]

	net, mean = vgg.net(data_path, x_reconstr)
	for layer in CONTENT_LAYERS:
		content_losses.append((2 * tf.nn.l2_loss(net[layer] - x_features[layer]) / size))

	content_loss = reduce(tf.add, content_losses)
	return content_loss

def content_loss(x, x_reconstr):
	def binary_crossentropy(t, o):
		return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))
	Lx = tf.reduce_sum(binary_crossentropy(x, x_reconstr), 1)  # reconstruction term
	Lx = tf.reduce_mean(Lx)

	return Lx

def style_loss(x, x_reconstr):
	style_features = {}
	net, mean = vgg.net(data_path, x)
	for layer in STYLE_LAYERS:
		feature = net[layer]
		size_origin = feature.get_shape().as_list()

		stderr.write('origin size is ' + str(size_origin) + '\n')
		feature = tf.reshape(feature, (-1, size_origin[3]))
		gram = tf.matmul(tf.transpose(feature), feature)
		style_features[layer] = gram

	x_grams = {}
	net, mean = vgg.net(data_path, x_reconstr)
	for layer in STYLE_LAYERS:
		feature = net[layer]
		size_origin = feature.get_shape().as_list()
		feature = tf.reshape(feature, (-1, size_origin[3]))
		gram = tf.matmul(tf.transpose(feature), feature)
		x_grams[layer] = gram

	Ls = 0
	i = 0
	all_weights = np.sum(omiga)
	for layer in STYLE_LAYERS:
		# stderr.write(str(reduce(mul, x_grams[layer].get_shape().as_list(), 1)))
		N2 = size_origin[3] * size_origin[3]
		M2 = size_origin[1] * size_origin[2] * size_origin[1] * size_origin[2]
		Ls += 2 * tf.nn.l2_loss(x_grams[layer] - style_features[layer]) / (4 * M2 * N2) * omiga[i] / all_weights
		i += 1

	return Ls

def seperate(x):
	return tf.transpose(x, perm=[0, 3, 1, 2])
def unseperate(x):
	return tf.transpose(x, perm=[0, 2, 3, 1])

# x: [batch, width, height, channel]
def reconstruct(x):
	global BUILT
	c = [0] * T
	mus, sigmas, logsigmas = [0] * T, [0] * T, [0] * T
	h_dec_prev = tf.zeros((1, dec_size), dtype=tf.float32)
	enc_state = encoder.zero_state(1, tf.float32)
	dec_state = decoder.zero_state(1, tf.float32)
	x = seperate(x) # x:[batch, channel, width, height]
	x = tf.reshape(x, [batch_size, -1])

	for t in range(T):
		c_prev = tf.zeros((batch_size, channel_ * width_ * height_)) if t == 0 else c[t-1]
		x_hat = x - tf.sigmoid(c_prev) # ?
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
	x_reconstr = unseperate(tf.reshape(x_reconstr, [batch_size, channel_, width_, height_]))

	Lz = 0
	for t in range(T):
		Lz += tf.square(mus[t]) + tf.square(sigmas[t]) + tf.square(logsigmas[t])
	Lz = 0.5 * ( tf.reduce_mean(Lz) - T )

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

def triple(Fx):
	shape = Fx.get_shape().as_list()
	res = tf.reshape(tf.tile(Fx, [shape[0], 3, 1]), [shape[0], channel, shape[1], shape[2]])
	return res # [channel, shape]

# x: [batch_size, -1], -1:channel * width * height
def read(x, x_hat, h_dec_prev, atten):
	if(atten == False):
		return tf.concat([x, x_hat], 1)
	else:
		Fx, Fy, gamma = attn_window("read", h_dec_prev, read_n)
		if(channel == 3):
			Fx = triple(Fx)
			Fy = triple(Fy)

		def filter_img(img, Fx, Fy, gamma, N):
			Fxt = tf.transpose(Fx, perm=[0, 1, 3, 2])
			img = tf.reshape(img, [-1, channel, width, height])
			temp = tf.matmul(img, Fxt)
			glimpse = tf.matmul(Fy, temp)
			glimpse = tf.reshape(glimpse, [-1, channel * N * N])
			return glimpse * tf.reshape(gamma, [-1, 1])

		x = filter_img(x, Fx, Fy, gamma, read_n)  # batch x (read_n*read_n)
		x = tf.reshape(x, [1, -1])
		# stderr.write(str(x.get_shape()))
		x_hat = filter_img(x_hat, Fx, Fy, gamma, read_n)
		x_hat = tf.reshape(x_hat, [batch_size, -1])
		# stderr.write(str(x_hat.get_shape()))
		return tf.concat([x, x_hat], axis=1)  # concat along feature axis


def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(width), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(height), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx,Fy


def attn_window(scope,h_dec,N):
    with tf.variable_scope(scope,reuse=BUILT):
        params=linear(h_dec,5)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(params, 5, axis=1)
    gx=(width+1)/2*(gx_+1)
    gy=(height+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(width,height)-1)/(N-1)*tf.exp(log_delta) # batch x N
    return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)


def write(h_dec, atten):
	if(atten == False):
		with tf.variable_scope("write", reuse=BUILT):
			return linear(h_dec, width_ * height_ * channel_)
	else:
		with tf.variable_scope("writeW", reuse=BUILT):
			w = linear(h_dec, channel_ * write_size)  # batch x (write_n*write_n)
		N = write_n
		w = tf.reshape(w, [batch_size, channel, N, N])
		Fx, Fy, gamma = attn_window("write", h_dec, write_n)
		# stderr.write(str(Fx.get_shape()))
		if(channel == 3):
			Fx = triple(Fx)
			Fy = triple(Fy)
		# 	# Fx = tf.reshape(tf.transpose(tf.tile(tf.reshape(Fx, [1, -1]), [3,1]), [1,0]), Fx.get_shape().as_list() + [channel])
		# 	# Fy = tf.reshape(tf.transpose(tf.tile(tf.reshape(Fy, [1, -1]), [3,1]), [1,0]), Fy.get_shape().as_list() + [channel])
		# 	# w = tf.reshape(tf.transpose(tf.tile(tf.reshape(w, [1, -1]), [3,1]), [1,0]), w.get_shape().as_list() + [channel])
		# 	Fx = tf.reshape(tf.tile(Fx, [3, 1, 1]), [batch_size, 3, Fx.get_shape().as_list()[1], Fx.get_shape().as_list()[2]])
		# 	Fy = tf.reshape(tf.tile(Fy,[3, 1, 1]), [batch_size, 3, Fy.get_shape().as_list()[1], Fy.get_shape().as_list()[2]])
		# 	w = tf.reshape(tf.tile(w, [3, 1, 1]), [batch_size, 3, w.get_shape().as_list()[1], w.get_shape().as_list()[2]])
		Fyt = tf.transpose(Fy, perm=[0, 1, 3, 2])
		wr = tf.matmul(Fyt, tf.matmul(w, Fx))
		wr = tf.reshape(wr, [batch_size, channel_ * width_ * height_])
		# wr = tf.reshape(tf.transpose(wr, perm=[0, 2, 1]), [batch_size, height * width * channel])
		# gamma=tf.tile(gamma,[1,height * width * channel])
		return wr * tf.reshape(1.0 / gamma, [-1, 1])


# train the network
def train():
	with tf.Session() as sess:
		image = tf.placeholder(tf.float32, shape = [width, height])


# generate a new picture
def generate():
	return


if __name__ == '__main__':
	main()
