# -*- coding:utf-8 -*-
import os
import os.path
import numpy as np
import tensorflow as tf
import input_trainval
import model_structure
import tools

# 参数设置 #
IMG_W = 400
IMG_H = 400
N_CLASSES = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MAX_STEP = 40
CAPACITY = 256
RATION = 0.2

# 训练 #
def train():
    data_dir = '/home/xinlong/Tensorflow_workspace/canjian_AlexNet/JPG/trainval/'
    train_log_dir = '/home/xinlong/Tensorflow_workspace/canjian_AlexNet/log/train/'
    val_log_dir = '/home/xinlong/Tensorflow_workspace/canjian_AlexNet/log/val/'

    with tf.name_scope('input'):
        train, train_label, val, val_label = input_trainval.get_files(data_dir, 0.2)
        train_batch, train_label_batch = input_trainval.get_batch(train, train_label,
                                                                  IMG_H, IMG_W,
                                                                  BATCH_SIZE,
                                                                  CAPACITY)
        val_batch, val_label_batch = input_trainval.get_batch(val, val_label,
                                                              IMG_W, IMG_H,
                                                              BATCH_SIZE,
                                                              CAPACITY)

        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_H, IMG_W, 3])
        y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        logits = model_structure.AlexNet(x, 5)
        loss = tools.loss('loss', y_, logits)
        accuracy = tools.accuracy('accuracy', y_, logits)

        my_global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tools.optimize('optimize', loss, LEARNING_RATE, my_global_step) #??

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()

        init = tf.initialize_all_variables()


        with tf.Session() as sess:
            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            tra_summary_writer = tf.summary.FileWriter(train_log_dir,sess.graph)
            val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

            try:
                for step in np.arange(MAX_STEP):
                    if coord.should_stop():
                        break

                    tra_images, tra_labels = sess.run([train_batch, train_label_batch])

                    _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],feed_dict={x:tra_images, y_:tra_labels})

                    if step % 10 == 0 or (step + 1) == MAX_STEP:
                        print('Step: %d, loss: %.4f, accuracy: %.4f' %(step, tra_loss, tra_acc))

                        #summary_str = sess.run(summary_op)

                        #tra_summary_writer.add_summary(summary_str, step)
                        checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                    
                    if step % 20 == 0 or (step + 1) == MAX_STEP:
                        valid_images, valid_labels = sess.run([val_batch, val_label_batch])
                        valid_loss, valid_acc = sess.run([loss, accuracy],
                                                         feed_dict={x:valid_images, y_:valid_labels})
                        print( '** step: %d,  loss: %.4f,  accuracy: %.4f' %(step, valid_loss, valid_acc))
                        #summary_str = sess.run(summary_op)
                        #val_summary_writer.add_summary(summary_str, step)


                    if step % 2000 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)


            except tf.error.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)


train()







