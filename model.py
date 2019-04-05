"""
@author:  wangquaxiu
@time:  2018/9/16 21:28

卷积神经网络模型


"""

import tensorflow as tf

def inference(images, batch_size, n_class):
    '''Build the model
       Args:
           images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
       Returns:
           output tensor with the computed logits, float, [batch_size, n_classes]


       '''
    # conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    # 声明第一层卷积层的变量并实现前向传播过程。
    # 通过使用不同的命名空间来隔离不同层的变量，这可以让每一层的变量命名只需要
    # 考虑当前层的作用，而不需要担心重名的问题。
    with tf.variable_scope('conv1') as scope:
        # 第一层卷积层的尺寸和深度。
        CONV1_DEEP = 16  # 既是过滤器的深度，又是下一层节点矩阵的深度
        CONV1_SIZE = 3
        NUM_CHANNELS = 3 # 当前层的深度
        # 过滤器的权重变量
        weights = tf.get_variable('weights',
                                  shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.1, dtype=tf.float32))

        # 过滤器的偏置项变量
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # 定义第一层池化层
    # 池化层可以有效的缩小矩阵的尺寸，从而减少最后全连接层的参数。
    with tf.variable_scope('pooling1_lrn') as scope:
        # 最大池化层的前向传播算法。
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')


    # conv2
    # 第二层卷积层
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,16,16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')



    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')


    # local3 全连接层
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)



    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # softmax层
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_class],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_class],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]


    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss


def training(loss, learning_rate):
    '''
    Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label

    :param logits: logits tensor, float - [batch_size, NUM_CLASS].
    :param labels: labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASS]

    :return:
        A scalar int32 tebnsor with the number of example (out of batch_size)
        that were predicted correctly.

    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy


