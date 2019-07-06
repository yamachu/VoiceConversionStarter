import tensorflow as tf

# Constant
input_size = 177 # Passed feature size
output_size = 177 # Number of outputs
hidden_size = 500

# Todo: Batch input...

tf.set_random_seed(1)

X = tf.placeholder(tf.float32, [None, input_size], name="X")
Y = tf.placeholder(tf.float32, [None, output_size], name="Y")
lr = tf.placeholder(tf.float32, name = "learning_rate")

h1 = tf.layers.dense(inputs=X,units=hidden_size,activation=tf.nn.leaky_relu)
h2 = tf.layers.dense(inputs=h1,units=hidden_size,activation=tf.nn.leaky_relu)
h3 = tf.layers.dense(inputs=h2,units=hidden_size,activation=tf.nn.leaky_relu)

W = tf.Variable(tf.random_normal([hidden_size,output_size], stddev=0.1), name = 'W4')
b = tf.Variable(tf.zeros([output_size]), name = 'b4')

Converted = tf.add(tf.add(tf.matmul(h3, W), b), X, name = 'Converted')

diff = tf.subtract(Converted, Y)
loss = tf.nn.l2_loss(diff)
loss = tf.identity(loss, name="Loss")
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Optimizer').minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    tf.saved_model.simple_save(sess, r'../Models/McepNN/model', inputs={'X': X, 'Y': Y}, outputs={'Converted': Converted} )
