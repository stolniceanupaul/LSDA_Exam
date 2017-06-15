# MLP
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Splits data into mini-batches
# Batches are not randomized/shuffled, shuffling the data in mini-batch learning typically improves the performance
class Batcher:
    'Splits data into mini-batches'
    def __init__(self, data, batchSize):
        self.data = data
        self.batchSize = batchSize
        self.batchStartIndex = 0
        self.batchStopIndex = 0
        self.noData = self.data.data.shape[0]
    def nextBatch(self):
        self.batchStartIndex = self.batchStopIndex % self.noData
        self.batchStopIndex = min(self.batchStartIndex + self.batchSize, self.noData)
        return self.data.data[self.batchStartIndex:self.batchStopIndex], self.data.target[self.batchStartIndex:self.batchStopIndex]

# Flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', 'tmp/MLPMiniLog', 'directory to put the summary data')
flags.DEFINE_string('data_dir', 'data/', 'directory with data')
flags.DEFINE_integer('maxIter', 10000, 'number of iterations')
# Batch size changed to 128
flags.DEFINE_integer('batchSize', 128, 'batch size')
flags.DEFINE_integer('noHidden1', 64, 'size of first hidden layer')
flags.DEFINE_integer('noHidden2', 32, 'size of second hidden layer')
flags.DEFINE_float('lr', 0.001, 'initial learning rate')

# Read data
# Changed to set the input data from the Galaxies dataset
# Changed the data type for the output variable - np.float32
dataTrain = tf.contrib.learn.datasets.base.load_csv_without_header(filename=FLAGS.data_dir + 'LSDA2017GalaxiesTrain.csv', target_dtype=np.float32, features_dtype=np.float32, target_column=-1)
dataTest = tf.contrib.learn.datasets.base.load_csv_without_header(filename=FLAGS.data_dir + 'LSDA2017GalaxiesTest.csv', target_dtype=np.float32, features_dtype=np.float32, target_column=-1)
dataValidate = tf.contrib.learn.datasets.base.load_csv_without_header(filename=FLAGS.data_dir + 'LSDA2017GalaxiesValidate.csv', target_dtype=np.float32, features_dtype=np.float32, target_column=-1)

# Number of training data points
noTrain = dataTrain.data.shape[0]
print("Numer of training data points:", noTrain)

# Input dimension
inDim = dataTrain.data.shape[1]

# Create graph
sess = tf.Session()

# Initialize placeholders
x_data = tf.placeholder(shape=[None, inDim], dtype=tf.float32, name='input')
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='target')

# Define variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # restrict to +/- 2*stddev
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='bias')


# Define model
# This is the 1st hidden layer with 64 neurons, connected to the input layer
with tf.name_scope('layer1') as scope:
    W_1 = weight_variable([inDim, FLAGS.noHidden1])
    b_1 = bias_variable([FLAGS.noHidden1])
    y_1 = tf.nn.sigmoid(tf.matmul(x_data, W_1) + b_1)

# This is the 2nd hidden layer with 32 neurons, connected to the 1st hidden layer
with tf.name_scope('layer2') as scope:
    W_2 = weight_variable([FLAGS.noHidden1, FLAGS.noHidden2])
    b_2 = bias_variable([FLAGS.noHidden2])
    y_2 = tf.nn.sigmoid(tf.matmul(y_1, W_2) + b_2)

# This is the output layer, connected to the 2nd hidden layer
with tf.name_scope('layer3') as scope:
    W_3 = weight_variable([FLAGS.noHidden2, 1])
    b_3 = bias_variable([1])
    model_output = tf.matmul(y_2, W_3) + b_3


# Declare loss function
# Use the mean squared-error as error function
loss = tf.reduce_mean(tf.squared_difference(x=model_output, y=y_target), name='squared_difference')
tf.summary.scalar('mean_squared_error', loss)

# Declare optimizer
my_opt =  tf.train.AdamOptimizer(FLAGS.lr)
train_step = my_opt.minimize(loss)

# Logging
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train')
test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')
validate_writer=  tf.summary.FileWriter(FLAGS.summary_dir + '/validate')
writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
saver = tf.train.Saver() # for storing the best network

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Initialize the bestValidation error with infinit
# It will be used to keep track of the minimum validation error
bestValidation = np.inf

# Mini-batches for training
batcher = Batcher(dataTrain, FLAGS.batchSize)

# Training loop
for i in range(FLAGS.maxIter):
    xTrain, yTrain = batcher.nextBatch()
    sess.run(train_step, feed_dict={x_data: xTrain, y_target: np.transpose([yTrain])})
    summary = sess.run(merged, feed_dict={x_data: xTrain, y_target: np.transpose([yTrain])})
    train_writer.add_summary(summary, i)
    if((i+1)%100==0):
        # print("Iteration:",i+1,"/",FLAGS.maxIter)
        summary = sess.run(merged, feed_dict={x_data: dataTest.data, y_target: np.transpose([dataTest.target])})
        test_writer.add_summary(summary, i)
        currentValidation, summary = sess.run([loss, merged], feed_dict={x_data: dataValidate.data, y_target: np.transpose([dataValidate.target])})
        validate_writer.add_summary(summary, i)
        if(currentValidation < bestValidation):
            bestValidation = currentValidation
            saver.save(sess=sess, save_path=FLAGS.summary_dir + '/bestNetwork')
            print("\tbetter network stored, old mse ",currentValidation,"< new mse ",bestValidation)

# Print values after last training step
print("final training error:", sess.run(loss, feed_dict={x_data: dataTrain.data, y_target: np.transpose([dataTrain.target])}))
print("final test error: ", sess.run(loss, feed_dict={x_data: dataTest.data, y_target: np.transpose([dataTest.target])}))
print("final validation error: ", sess.run(loss, feed_dict={x_data: dataValidate.data, y_target: np.transpose([dataValidate.target])}))

# Load the network with the lowest validation error
saver.restore(sess=sess, save_path=FLAGS.summary_dir + '/bestNetwork')
print("best training error:", sess.run(loss, feed_dict={x_data: dataTrain.data, y_target: np.transpose([dataTrain.target])}))
print("best test error: ", sess.run(loss, feed_dict={x_data: dataTest.data, y_target: np.transpose([dataTest.target])}))
print("best validation error: ", sess.run(loss, feed_dict={x_data: dataValidate.data, y_target: np.transpose([dataValidate.target])}))
