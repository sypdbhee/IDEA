from __future__ import division
import tensorflow as tf

input_dim = 257

def conv(inputs, dim, name, training):
    outputs=tf.layers.conv1d(inputs, dim, 32, 1, padding='SAME')
    outputs=tf.nn.relu(outputs, name=name)
    return outputs

class CNN:
    def __init__(self):
        self._generate = tf.make_template('G', self._generator)

    def _generator(self, x, training=False):
        with tf.name_scope('G'):
            h1 = conv(x, 32, 'c1', training)
            h2 = conv(h1, 32, 'c2', training)
            h3 = tf.contrib.layers.flatten(h2)
            outputs = tf.layers.dense(h3, 2048)
            outputs=tf.nn.relu(outputs)
            outputs = tf.layers.dense(outputs, input_dim, name='outputs')
        return outputs

    def predict(self, x):
        with tf.name_scope('predict'):
            pred = self._generate(x)
        return pred

    def loss(self, x, y):
        with tf.name_scope('loss'):
            pred = self._generate(x, training=True)
            loss = self.mse(pred, y)
            tf.summary.scalar('loss', loss)
        return loss

    def generate(self, x):
        return self._generator(x)

    def mse(self, x, y):
        return tf.reduce_mean(tf.squared_difference(x,y))


