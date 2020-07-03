import time, os, pickle
import math, random
import numpy as np
import tensorflow as tf
from readgraph import ImportGraph
from ensemble import CNN

input_dim = 257
iter_num = 100
batch_num = 128

def next_batch(x):
    data = []
    for i in range(5, len(x)-5):
        data.append(x[i-5:i+6, :])
    return np.asarray(data)

def s_train(data, path):
    start = time.time()
    x_train, y_train = data['x'], data['y']
    
    x = tf.placeholder(tf.float32, [None, input_dim, 3], name='x')
    y = tf.placeholder(tf.float32, [None, input_dim], name='y')
    os.system("mkdir -p " + path)

    model = CNN()
    loss = model.loss(x, y)
    pred = model.predict(x)
    optimize =  tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.9, beta2=0.999).minimize(loss)
    merged = tf.summary.merge_all() 

    g1 = ImportGraph('./3/')
    g2 = ImportGraph('./6/')
    g3 = ImportGraph('./9/')
    
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('pred', pred)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(path+'logs', sess.graph)
        for i in range(iter_num):
            for j in range(len(x_train)):
                idx = random.randint(0, len(x_train)-1)
                xt = next_batch(x_train[idx])
                out_1 = g1.pred(xt)
                out_2 = g2.pred(xt)
                out_3 = g3.pred(xt)
                xt = np.dstack((out_1, out_2, out_3))
                yt = y_train[idx%len(y_train)][5:-5, :]
                k = 0
                for k in range(0, len(xt), batch_num):
                    xb = xt[k:k+batch_num]
                    yb = yt[k:k+batch_num]
                    _ = sess.run([optimize], feed_dict={x:xb, y:yb})
                xb = xt[k:]
                yb = yt[k:]
                err, result = sess.run([loss, merged], feed_dict={x:xb, y:yb})
                if j%100 == 0:
                    writer.add_summary(result, len(x_train)*i+j)
                    saver.save(sess, path+'test_best_model')
                    print('Epoch [%4d] Iter [%4d] Time [%5.4f] \nLoss: [%.4f]' %
                      (i+1, j, time.time() - start, err))


def s_test(spec, path, data):
    x = spec['x']
    y = spec['y']
    os.system("mkdir -p "+path+data)

    g1 = ImportGraph('./3/')
    g2 = ImportGraph('./6/')
    g3 = ImportGraph('./9/')
    graph = ImportGraph(path)
    preds = []

    for i in range(len(x)):
        xb = next_batch(x[i])
        yb = y[i%len(y)][5:-5, :]
        out1  = g1.pred(xb)
        out2  = g2.pred(xb)
        out3  = g3.pred(xb)
        out = np.dstack((out1, out2, out3))
        pred = graph.pred(out)
        preds.append(pred)
        loss = graph.loss(out, yb)
        print('Loss: %.4f' % loss)
    with open(path+data+'/pred.txt', 'wb') as fp:
        pickle.dump(preds, fp)

