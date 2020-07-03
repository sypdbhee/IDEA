import tensorflow as tf

class ImportGraph():
    def __init__(self, path):
        self.path = path
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.path+'test_best_model.meta', clear_devices=True)
            saver.restore(self.sess, self.path+'test_best_model')
            self.predict=tf.get_collection('pred')[0]
            self.err = tf.get_collection('loss')[0]

    def pred(self, data):
        return self.sess.run(self.predict, feed_dict={'x:0': data})

    def loss(self, x, y):
        return self.sess.run(self.err, feed_dict={'x:0': x, 'y:0':y})
