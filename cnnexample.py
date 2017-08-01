import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#10 classes, 0-9

n_classes = 10
batch_size= 128


#x is the data, y is the label of that data 
#tf Graph input
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

#define a couple super simple functions that will help us with our convolutions and pooling:
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
'''
tf.nn.conv2d:
这个函数的功能是：给定4维的iput和filter，计算出一个2维的卷积结果：
def conv2d(input, filter, strides, padding, use_cudnn_ongpu=None, data_format=None, name=None):
    input：待卷积的数据。格式要求为一个张量，[batch, in_height, in_width, in_channels]. 
    分别表示 批次数，图像高度，宽度，输入通道数。 
    filter： 卷积核。格式要求为[filter_height, filter_width, in_channels, out_channels]. 
    分别表示 卷积核的高度，宽度，输入通道数，输出通道数。 
    strides :一个长为4的list. 表示每次卷积以后卷积窗口在input中滑动的距离 [batch, height, width, channels]
    padding ：有SAME和VALID两种选项，表示是否要保留图像边上那一圈不完全卷积的部分。如果是SAME，则保留 
    use_cudnn_on_gpu ：是否使用cudnn加速。默认是True
'''
def maxpool2d(x):
    #                         size of window     movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
'''
tf.nn.max_pool :

进行最大值池化操作,而avg_pool 则进行平均值池化操作.函数的定义为：

def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):

value: 一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样 
ksize: 长为4的list,表示池化窗口的尺寸 [batch, height, width, channels]
strides: 池化窗口的滑动值，与conv2d中的一样  [batch, height, width, channels]
padding: 与conv2d中用法一样。


那么这里就是x为value即我们的dataset，
'''




#现在我们来构建我们的CNN网络结构
def convolutional_neural_network_model(x):
    #参数构建：
    #filter： 卷积核。格式要求为[filter_height, filter_width, in_channels, out_channels].
    weights = {
                #5 X 5 convolution, 1 input image, 32 outputs
               'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
               #5 X 5 convolution, 32 inputs, 64 outputs
               'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
               #fully connected, 7*7*64 inputs
               'W_fc': tf.Variable(tf.random_normal([7*7*64,1024])),
               # fully connected, 7*7*64 inputs, 1024 outputs
               'out': tf.Variable(tf.random_normal([1024, n_classes])),}

    #we also have a bias vector with a component for each output channel
    #所以，这里biases的值是输出的大小
    biases = { 'b_conv1': tf.Variable(tf.random_normal([32])),
               'b_conv2': tf.Variable(tf.random_normal([64])),
               'b_fc': tf.Variable(tf.random_normal([1024])),
               'out': tf.Variable(tf.random_normal([n_classes])),}
'''
在这里说明一下，发生了什么。
第一层：我们使用5X5的卷积核在最初的图像上，产生了32个ouputs（feature maps），经过第一层的pooling（2X2）之后，
变成了14X14X32的输出。
第二层：再使用5X5的卷积在32的input中，产生了64个outputs（feature maps），经过第二层的pooling（2X2），之后
变成了7X7X64的输出。
第三层：7X7X64经过1024的神经元进行运输，所以output是1024。
第四层：1024经过softmax层进行n_classes的预测。
'''





#模型构建：
    #把图片变成28*28的图片（reshape我们的input变成4D tensor:[batch, height, width, channels]）
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    '''
    x:input：待卷积的数据。格式要求为一个张量，[batch, in_height, in_width, in_channels]. 
    分别表示 批次数，图像高度，宽度，输入通道数。 
    '''
    #convolution layer，使用我们的conv2d函数
    conv1 = conv2d(x, weights['W_conv1'])
    #maxpooling （down—sampling）使用maxpool2d函数
    conv1 = maxpool2d(conv1)
    #convolution layer，使用我们的conv2d函数
    conv2 = conv2d(conv1, weights['W_conv2'])
    #maxpooling （down—sampling）使用maxpool2d函数
    conv2 = maxpool2d(conv2)

    #Fully connected layer
    #Reshape conv2 output to fit fully connected layer
    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']  

    return output






def train_neural_network(x):
    prediction = convolutional_neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits =prediction,labels =y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 15

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y=mnist.train.next_batch(batch_size)
                _, c=sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Echoch', epoch, 'completed out of', hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))





train_neural_network(x)
