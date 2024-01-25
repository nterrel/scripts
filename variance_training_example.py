import matplotlib.pyplot as plt
import random
import math
import numpy as np
import tensorflow as tf
import sys

print('Python {}'.format(sys.version))
print('Tensorflow version {}'.format(tf.__version__))

# make some fake data:
inc = 0.001
x_train = np.concatenate([np.arange(-2, -1.5, inc),
                          np.arange(-1, 2, inc)])
x_train = x_train.reshape(len(x_train), 1)

steps_per_cycle = 1


def sinfun(xs, noise=0.001):
    xs = xs.flatten()

    def randomNoise(x):
        ax = 2 - np.abs(x)
        wnoise = random.uniform(-noise * ax,
                                 noise * ax)
        return (math.sin(x * (2 * math.pi / steps_per_cycle)) + wnoise)
    vec = [randomNoise(x) - x for x in xs]
    return (np.array(vec).flatten())


y_train0 = sinfun(x_train, noise=0.5)
y_train = y_train0.reshape(len(y_train0), 1)

print('x_train.reshape={}'.format(x_train.shape))
print('y_train.reshape={}'.format(y_train.shape))

plt.figure(figsize=(10, 3))
plt.scatter(x_train, y_train, s=0.5)
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.title('The x values of the synthetic data ranges between {:4.3f} and {:4.3f}'.format(
    np.min(x_train), np.max(x_train)))
plt.show()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01, dtype='float32')
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape, dtype='float32')
    return tf.Variable(initial)


def fully_connected_layer(h0, n_h0, n_h1, verbose=True):
    '''
    h0  :   tensor of shape (n_h0, n_h1)
    n_h0:   scalar
    n_h1:   scalar
    '''
    w1 = weight_variable([n_h0, n_h1])
    b1 = bias_variable([n_h1])

    if verbose:
        print('h0.shape={}'.format(h0.get_shape()))
        print('w1.shape={}'.format(w1.get_shape()))
        print('b1.shape={}'.format(b1.get_shape()))

    h1 = tf.matmul(h0, w1) + b1
    return (h1, (w1, b1))


def nll_gaussian(y_pred_mean, y_pred_stdev, y_test):
    square = tf.square(y_pred_mean - y_test)
    ms = tf.add(tf.divide(square, y_pred_stdev), tf.log(y_pred_stdev))
    ms = tf.reduce_mean(ms)
    return (ms)


def mse(y_pred, y_test, verbose=True):
    square = tf.square(y_pred - y_test)
    ms = tf.reduce_mean(square)
    return (ms)


def define_model(n_feature, n_hs, n_output, eps=0.1, verbose=True, NLL=True):
    x_input_shape = [None, n_feature]
    x_input = tf.placeholder(tf.float32, x_input_shape)
    y_input = tf.placeholder(tf.float32, [None, 1])
    h_previous = x_input
    n_h_previous = n_feature
    paras = []
    for ilayer, n_h in enumerate(n_hs, 1):
        if verbose:
            print('layer:{}'.format(ilayer))
        h, p = fully_connected_layer(h_previous, n_h_previous, n_h, verbose)
        h_previous = tf.nn.relu(h)
        n_h_previous = n_h
        paras.append(p)
    if verbose:
        print('output layer for y_mean')
    y_mean, p = fully_connected_layer(h_previous, n_h_previous, n_output, verbose)
    paras.append(p)

    if NLL:
        if verbose:
            print('output layer for y_sigma')
        y_sigma, p = fully_connected_layer(h_previous, n_h_previous, n_output, verbose)
        y_sigma = tf.clip_by_value(t=tf.exp(y_sigma),
                                   clip_value_min=tf.constant(1E-1),
                                   clip_value_max=tf.constant(1E+100))
        paras.append(p)
        loss = nll_gaussian(y_mean, y_sigma, y_input)
        y = [y_mean, y_sigma]
    else:
        loss = mse(y_mean, y_input)
        y = [y_mean]

    eps_tf = tf.constant(float(eps), name='epsilon')
    grad_tf = tf.gradients(loss, [x_input])
    grad_sign_tf = tf.sign(grad_tf)
    grad_sign_eps_tf = tf.scalar_mul(eps_tf,
                                    grad_sign_tf)
    aimage = tf.add(grad_sign_eps_tf, x_input)
    aimage = tf.reshape(aimage, [-1, n_feature])
    x_input_a = tf.concat([aimage, x_input], axis=0)

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    inputs = (x_input, y_input, x_input_a)
    tensors = (inputs, loss, train_step, y, paras)
    return tensors
