import tensorflow as tf

a = tf.Variable(2.0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    nilai_a = sess.run(a)
    print(nilai_a)