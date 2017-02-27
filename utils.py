import scipy.misc


def imread(name, shape):
    return scipy.misc.imread(name, flatten=True)


def imsave(name, array):
    return scipy.misc.imsave(name, array)


def imresize(img, shape):
    return scipy.misc.imresize(img, shape, interp="bilinear")