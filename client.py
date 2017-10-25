# coding=utf-8
import tensorflow as tf
import numpy as np
import sys
import os

data = u'去年今日此门中，人面桃花相映红，人面不知何处去，桃花依旧笑春风。'
# data = u'人生若只如初见，何事秋风悲画扇。等闲变却故人心，却道故人心易变。骊山语罢清宵半，泪雨零铃终不怨。何如薄幸锦衣郎，比翼连枝当日愿。'
# 开始/结束标识位
START, END = 'S', 'E'
# 模型参数保存路径
MODEL_PATH = 'model_path/'
# 最大迭代次数
MAX_ITERATOR = 1000
# 是否输出数据信息
OUTPUT_DATA_INFO = True
# 每步预测都输出
OUTPUT_EVERY_STEP = False
# 训练
TRAIN = True
# 预测
PREDICT = True


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


def inference(x, batch_size):
    n_hidden = 32
    depth = int(x.get_shape()[-1])

    x = tf.reshape(x, [-1, depth])

    W = weight([depth, n_hidden])
    b = bias([n_hidden])
    x = tf.matmul(x, W) + b

    x = tf.reshape(x, [batch_size, -1, n_hidden])

    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)
    x = tf.reshape(outputs, [-1, n_hidden])

    W = weight([n_hidden, depth])
    b = bias([depth])
    x = tf.matmul(x, W) + b

    return x, initial_state, last_state


def train_info(logits, labels, learning_rate=1e-2):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return loss, optimizer


def p(array):
    if not OUTPUT_DATA_INFO:
        return
    r = ''
    if isinstance(array, dict):
        array = ['%s:%s ' % (k, v) for k, v in array.items()]
    for a in array:
        if isinstance(a, int):
            a = str(a) + ' '
        if isinstance(a, list):
            a = str(a) + ' '
        r += a
    print '[%s] %s' % (len(array), r)


def word2onehot(w, words2index, depth):
    index = words2index[w]
    return [1 if index == i else 0 for i in range(depth)]


data = START + data + END

words = list(set(data))
depth = len(words)
index2words = {i: word for i, word in enumerate(words)}
words2index = {v: k for k, v in index2words.items()}
p(words)
p(index2words)
p(words2index)

x_data = data
y_data = [x_data[i + 1] if i < len(x_data) - 1 else END for i in range(len(x_data))]
p(x_data)
p(y_data)

x_data = list(map(lambda w: word2onehot(w, words2index, depth), x_data))
y_data = list(map(lambda w: word2onehot(w, words2index, depth), y_data))
p(x_data)
p(y_data)

x = tf.placeholder(tf.float32, [None, depth])
y = tf.placeholder(tf.float32, [None, depth])

logits, initial_state, last_state = inference(x, 1)
loss, train_step = train_info(logits, y)

if not TRAIN and not PREDICT:
    exit()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint(MODEL_PATH)
if checkpoint:
    saver.restore(sess, checkpoint)
    print 'Load last model params successfully.'

if TRAIN:
    for step in range(1, MAX_ITERATOR + 1):
        loss_value, _ = sess.run([loss, train_step], feed_dict={x: x_data, y: y_data})
        if step % 10 == 0 or step == MAX_ITERATOR:
            print 'Step %s, Loss %s' % (step, loss_value)
        if step % 50 == 0 or step == MAX_ITERATOR:
            saver.save(sess, os.path.join(MODEL_PATH, 'model'))
if PREDICT:
    inputs = [START]
    inputs = list(map(lambda w: word2onehot(w, words2index, depth), inputs))
    predict = tf.nn.softmax(logits)
    outputs, last_state_value = sess.run([predict, last_state], feed_dict={x: inputs})
    word = index2words[np.argmax(outputs)]
    result = ''
    while word != END:
        result += word
        if OUTPUT_EVERY_STEP:
            sys.stdout.write(word)
            sys.stdout.flush()
        outputs, last_state_value = sess.run(
            [predict, last_state], feed_dict={x: outputs, initial_state: last_state_value})
        word = index2words[np.argmax(outputs)]
    print '\n[Output]'
    print result
