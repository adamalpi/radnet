import tensorflow as tf


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x


def pool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def ReLU(x):
    # return tf.nn.relu(x)
    return leakyReLU(x, 0.001)


def Sigmoid(x):
    return tf.nn.sigmoid(x)


def weightInitilization5(a, b, c, d, wstddev):
    return tf.Variable(tf.random_normal([a, b, c, d], stddev=wstddev))


def weightInitilization3(a, b, wstddev):
    return tf.Variable(tf.random_normal([a, b], stddev=wstddev))

# in the lecture 5 slide 38 set b to small value i.e. 0.1
def biasInitialization(a, bstddev):
    return tf.Variable(tf.random_normal([a], stddev=bstddev, mean=0.1))
    # return tf.Variable(tf.zeros([a]))


# https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/V6aeBw4nlaE
def leakyReLU(x, alpha=0., max_value=None):
    '''Rectified linear unit
    # Arguments
        alpha: slope of negative section.
        max_value: saturation threshold.
    '''

    if alpha != 0.:
        negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        max_value = tf.cast(max_value, x.dtype.base_dtype)
        zero = tf.cast(0., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, max_value)
    if alpha != 0.:
        x -= alpha * negative_part
    return x

c1_size = 32
c2_size = 64
c3_size = 128
fc1_size = 512
fc2_size = 256
out_size = 26
weight_stddev = 0.3
bias_stddev = 0.03

class RadNetModel(object):
    '''Implements the Radiation model for Climate Science

    TODO: Usage...

    '''

    def __init__(self):
        ''' Initializes the RadNet Model. '''
        self.vars = self._create_variables()


    def _create_variables(self):
        var = dict()

        with tf.variable_scope('radnet'):
            with tf.variable_scope('conv1'):
                current = dict()
                current['w'] = weightInitilization5(2, 2, 1, c1_size, weight_stddev)
                current['b'] = biasInitialization(c1_size, bias_stddev)
                var['conv1'] = current
            with tf.variable_scope('conv2'):
                current = dict()
                current['w'] = weightInitilization5(2, 2, c1_size, c2_size, weight_stddev)
                current['b'] = biasInitialization(c2_size, bias_stddev)
                var['conv2'] = current
            with tf.variable_scope('conv3'):
                current = dict()
                current['w'] = weightInitilization5(2, 2, c2_size, c3_size, weight_stddev)
                current['b'] = biasInitialization(c3_size, bias_stddev)
                var['conv3'] = current
            with tf.variable_scope('fc1'):
                current = dict()
                current['w'] = weightInitilization3(2 * 2 * c3_size, fc1_size, weight_stddev)
                current['b'] = biasInitialization(fc1_size, bias_stddev)
                var['fc1'] = current
            with tf.variable_scope('fc2'):
                current = dict()
                current['w'] = weightInitilization3(fc1_size, fc2_size, weight_stddev)
                current['b'] = biasInitialization(fc2_size, bias_stddev)
                var['fc2'] = current
            with tf.variable_scope('out'):
                current = dict()
                current['w'] = weightInitilization3(fc2_size, out_size, weight_stddev)
                current['b'] = biasInitialization(out_size, bias_stddev)
                var['out'] = current

        return var

    def _create_network(self, input_batch):
        ''' Construct the network.'''

        # Pre-process the input
        # x is 64 x 1 tensor with padding at the end
        input_batch = tf.reshape(input_batch, shape=[-1, 8, 8, 1])

        with tf.name_scope('conv1'):
            conv1 = conv2d(input_batch, self.vars['conv1']['w'], self.vars['conv1']['b'], strides=1)
            conv1 = ReLU(conv1)
            conv1 = pool2d(conv1, k=1)
        with tf.name_scope('conv2'):
            conv2 = conv2d(conv1, self.vars['conv2']['w'], self.vars['conv2']['b'], strides=1)
            conv2 = ReLU(conv2)
            conv2 = pool2d(conv2, k=2)
        with tf.name_scope('conv3'):
            conv3 = conv2d(conv2, self.vars['conv3']['w'], self.vars['conv3']['b'], strides=1)
            conv3 = ReLU(conv3)
            conv3 = pool2d(conv3, k=2)
        with tf.name_scope('fc1'):
            # Reshape conv3 output to fit fully connected layer input
            fc1 = tf.reshape(conv3, [-1, self.vars['fc1']['w'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.vars['fc1']['w']), self.vars['fc1']['b'])
            fc1 = ReLU(fc1)
        with tf.name_scope('fc2'):
            # Reshape conv3 output to fit fully connected layer input
            fc2 = tf.add(tf.matmul(fc1, self.vars['fc2']['w']), self.vars['fc2']['b'])
            fc2 = ReLU(fc2)
        with tf.name_scope('out'):
            out = tf.add(tf.matmul(fc2, self.vars['out']['w']), self.vars['out']['b'])

        return out

    def loss(self, input_batch, real_output):
        ''' Creates a RadNet network and returns the autoencoding loss.
            The variables are all scoped to the given name.
        '''
        with tf.name_scope('radnet'):
            output = self._create_network(input_batch)
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.squared_difference(output, real_output))
                tf.scalar_summary('loss', loss)

                return loss


    def predict(self, input, real_output, id_file):
        with tf.name_scope('radnet'):
            pred_output = self._create_network(input)
            loss = tf.reduce_mean(tf.squared_difference(pred_output, real_output))
            return id_file, real_output, pred_output, loss
