"""
@author:  wangquaxiu
@time:  2018/9/17 20:59
"""

import os
import numpy as np
import tensorflow as tf
import model
import create_records as cr

N_CLASSES = 2  # 分类数量
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.001 # with current parameters, it is suggested to use learning rate<0.0001


def run_training1():
    train_dir = 'G:/Python/dogorcat/data/train'
    tfrecords_dir = 'tfrecords/'
    tfrecords_file = 'test.tfrecords'
    logs_train_dir = 'logs/recordstrain/'
    # images, labels = cr.get_files(train_dir)
    # cr.convert_to_tfrecord(images, labels, tfrecords_dir, tfrecords_file)


    train_batch, train_label_batch = cr.read_and_decode(tfrecords_dir+tfrecords_file, batch_size=BATCH_SIZE)
    train_batch = tf.cast(train_batch, dtype=tf.float32)
    train_label_batch = tf.cast(train_label_batch, dtype=tf.int64)

    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, learning_rate)
    train_acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step+1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

run_training1()
