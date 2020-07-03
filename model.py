import time, os, pickle
import math, random
import numpy as np
import tensorflow as tf
from readgraph import ImportGraph
from dnn import DNN
import pdb

input_dim = 257
iter_num = 100
batch_num = 128

gpu_options=tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.45);
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options));

def next_batch(x):
    data = []
   # pdb.set_trace()
    for i in range(5, len(x)-5):
        data.append(x[i-5:i+6, :])
    return np.asarray(data)

def train(data, path):
    os.system("mkdir -p " + path)
    x_train, y_train = data['x'], data['y']
    #pdb.set_trace()
    x = tf.placeholder(tf.float32, [None, 11, input_dim], name='x')

    y = tf.placeholder(tf.float32, [None, input_dim], name='y')

    model = DNN()
    loss = model.loss(x, y)
    pred = model.predict(x)
    tf.add_to_collection('pred', pred)
    tf.add_to_collection('loss', loss)

    optimize = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.9, beta2=0.999).minimize(loss)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        start = time.time()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(path+'logs', sess.graph)

        err_old=100;
        for i in range(iter_num):
            err_new=0;
            count=0;
            for j in range(len(x_train)):
                idx = random.randint(0, len(x_train)-1)
                #pdb.set_trace()
                xt = next_batch(x_train[idx])
                yt = y_train[idx%len(y_train)][5:-5, :]
                k = 0
                for k in range(0, len(xt), batch_num):
                    xb = xt[k:k+batch_num]
                    yb = yt[k:k+batch_num]
                    _ = sess.run([optimize], feed_dict={x:xb, y:yb})
                xb = xt[k:]
                yb = yt[k:]
                err, result = sess.run([loss, merged], feed_dict={x:xb, y:yb})
                err_new+=err
                count+=1
                if j%100 == 0:
                    writer.add_summary(result, len(x_train)*i+j)
            if err_new/count < err_old:
                err_old=err_new/count
                saver.save(sess, path+'test_best_model')
                    #print('Epoch [%4d] Iter [%4d] Time [%5.4f] \nLoss: [%.4f]' %
                    #  (i+1, j, time.time() - start, err))
                print('Epoch [%4d] Time [%10.4f] Loss: [%.4f]: Saved ' %
                      (i+1, time.time() - start, err_new/count))
            else:
                print('Epoch [%4d] Time [%5.4f] Loss: [%.4f]: No save ' %
                      (i+1, time.time() - start, err_new/count))

def test(spec, path, data):
    x = spec['x']
    y = spec['y']
    os.system("mkdir -p "+path+data)

    graph = ImportGraph(path)
    preds = []

    for i in range(len(x)):
        xb = next_batch(x[i])
        yb = y[i%len(y)][5:-5, :]
        pred  = graph.pred(xb)
        preds.append(pred)
        loss = graph.loss(xb, yb)
        print('Loss: %.4f' % loss)
    with open(path+data+'/pred.txt', 'wb') as fp:
        pickle.dump(preds, fp)


