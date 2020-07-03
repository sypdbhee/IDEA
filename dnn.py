from __future__ import division
import tensorflow as tf

input_dim = 257

def nn(inputs, dim, name, training):
    outputs=tf.layers.dense(inputs, dim)
    #outputs = tf.layers.dropout(outputs, rate=0.35, training=training)
    outputs = tf.nn.relu(outputs, name=name)
    return outputs

class DNN:
    def __init__(self):
        self._generate = tf.make_template('G', self._generator)

    def _generator(self, x, training=False):
        with tf.name_scope('G'):
            inputs = tf.contrib.layers.flatten(x)
            h1 = nn(inputs, 2048, 'h1', training)
            h2 = nn(h1, 2048, 'h2', training)
            h3 = nn(h2, 2048, 'h3', training)
            #h4 = nn(h3, 2048, 'h4', training)
            #h5 = nn(h4, 2048, 'h5', training)
            #h6 = nn(h5, 2048, 'h6', training)
            h7 = tf.concat([h1, h3], axis=1)
            outputs = tf.layers.dense(h7, input_dim, name='outputs')
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

