import tensorflow as tf


tf.set_random_seed(1)

# Constant
input_size = 40 # Passed feature size
output_size = 40 # Number of outputs
hidden_size = 500

# Parameters
lr = tf.placeholder(tf.float32, name='learning_rate')

# Todo: Batch input...

# Graph Input
X = tf.placeholder(tf.float32, [None, input_size], name='X')
# Graph Label
Y = tf.placeholder(tf.float32, [None, output_size], name='Y')

h1 = tf.layers.dense(inputs=X, units=hidden_size, activation=tf.nn.leaky_relu)
h2 = tf.layers.dense(inputs=h1, units=hidden_size, activation=tf.nn.leaky_relu)
h3 = tf.layers.dense(inputs=h2, units=hidden_size, activation=tf.nn.leaky_relu)

W = tf.Variable(tf.random_normal([hidden_size, output_size], stddev=0.1), name='W4')
b = tf.Variable(tf.zeros([output_size]), name='b4')
Converted = tf.add(tf.add(tf.matmul(h3, W), b), X, name='Converted')

loss = tf.reduce_mean(tf.square(Y - Converted), name='Loss')
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, name='Optimizer')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    tf.saved_model.simple_save(
        sess,
        '../Models/McepNN/model',
        inputs={'X': X, 'Y': Y},
        outputs={'Converted': Converted}
    )
