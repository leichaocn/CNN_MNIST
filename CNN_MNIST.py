#引入mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

#定义权值初始化函数，初始化为一个截断正太分布
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
#定义偏置初始化函数
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#定义卷积初始化函数
#步长为1，补1圈零。
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#定义池化初始化函数，池化窗shape为2*2,步长为2，补1圈零。新的shape将是原shape边长/2+1
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
#定义两个占位符，x和y_用来存。                        
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


#第一次卷积+池化运算
#将向量x变成28*28的图片，深度为1
x_image = tf.reshape(x, [-1,28,28,1])
#定义Filter变量（即W）为[5,5]，深度为1（因为X_image深度为1），Filter个数为32个（由工程经验而定的超参数）。                        
W_conv1 = weight_variable([5, 5, 1, 32])
#定义偏执变量为32个
b_conv1 = bias_variable([32])
#进行卷积计算，x_image是28*28，W是5*5，卷积结果是(28-5+2)/1+1，26*26,深度是32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#进行池化计算，池化结果是14*14，深度是32
h_pool1 = max_pool_2x2(h_conv1)

#第二次卷积+池化运算
#Filter的shape是5*5，深度是32，个数是64（超参数）
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#进行卷积计算，h_pool1是14*14,深度32，卷积后（14-5+2）/1+1,12*12，深度为64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#池化后，h_pool2是7*7，深度是64
h_pool2 = max_pool_2x2(h_conv2)

#全连接层计算，使用dropout避免过拟合。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#将h_pool2的shape由立方体拉成一维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#计算完后，输出h_fc1的shape为1024的向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#输出层的计算
#由1024维，降为10维。
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#定义损失函数，即label与y_conv的交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
#定义优化器为Adam
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#定义准确率测试
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#初始化所有变量
tf.global_variables_initializer().run()

#运行计算图
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})#此处为输出训练结果，因此dropout为1
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})#此处为训练，故而dropout设为0.5


print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
