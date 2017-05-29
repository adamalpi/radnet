import tensorflow as tf


def conv2d(x, W, b, strides=1, padding='SAME'):
    """ Wrapper for using the applying a 2d conv with the tensorflow library
     and then add a bias.
    """
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return x


def pool2d(x, k=2, l=2):
    """ Wrapper for a max pool filter
    """
    return tf.nn.max_pool(x, ksize=[1, k, l, 1], strides=[1, k, l, 1], padding='SAME')


def ReLU(x, alpha):
    """ Wrapper for using a ReLU activation function
    TODO: Implement an option for choosing between relu, leaky relu and prelu.
    """
    # return tf.nn.relu(x)
    # return leakyReLU(x, 0.001)
    return parametricReLU(x, alpha)


def weightInitilization5(a, b, c, d, wstddev):
    """ Inits the weights for the convolutional layers. The initialization is using a
    normal distribution with a given standard deviation.

    :param a: height of filter
    :param b: width of filter
    :param c: in size
    :param d: out size
    :param wstddev: standard deviation
    :return:
    """
    return tf.Variable(tf.random_normal([a, b, c, d], stddev=wstddev))


def weightInitilization3(a, b, wstddev):
    """ Inits the wights for the fully connected layers.
    The the initialization is using xavier initialization that improves the convergence
    considerably.

    :param a: In size
    :param b: Out size
    :return:
    """
    # xavier initialization improves the starting of the training
    # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    return tf.get_variable("weight", shape=[a, b],
                        initializer=tf.contrib.layers.xavier_initializer())

    # earlier solution with random initialization.
    #return tf.Variable(tf.random_normal([a, b], stddev=wstddev))

def biasInitialization(a, bstddev):
    """ Initialization of the bias
    In the lecture 5 slide 38 set b to small value i.e. 0.1 of Jim's slides.

    :param a:
    :param bstddev:
    :return: x
    """
    return tf.Variable(tf.random_normal([a], stddev=bstddev, mean=0.1))
    # return tf.Variable(tf.zeros([a]))


def parametricReLU(x, alpha):
    """ PReLU
    Ref.: http://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
    Ref.: https://github.com/tflearn/tflearn/blob/4ba8c8d78bf1bbdfc595bf547bad30580cb4c20b/tflearn/activations.py#L191

    :param x:
    :param alpha: trainable variable for the negative values
    :return:
    """
    pos = tf.nn.relu(x)
    neg = alpha * (x - tf.abs(x)) * 0.5

    return pos + neg

def leakyReLU(x, alpha=0., max_value=None):
    """ leakyReLU

    # Ref.: https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/V6aeBw4nlaE

    :param alpha: slope of negative section. Hyperparameter
    :param max_value: saturation threshold.
    :return: x
    """

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


def bnInitialization(n_out):
    """ Batch normalization initialization of the variables of the layer.

    :param n_out:
    :return:
    """
    current = dict()
    with tf.variable_scope('bn'):
        current['beta'] = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta')

        current['gamma'] = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma')

        current['mean'] = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                trainable=False)
        current['var'] = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                    trainable=False)

        return current

def preluInitialization(n_out):
    """ Initialization of Alpha for the PReLU

    :param n_out:
    :return:
    """
    return tf.Variable(tf.constant(0.0, shape=[n_out]), name='alpha', dtype=tf.float32)


def batchNorm(x, axes, vars, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Ref.: https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
    Ref.: http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

    :param x: Tensor, 2d input maps
    :param n_out: integer, depth of input maps
    :param phase_train: boolean tf.Varialbe, true indicates training phase
    :param scope: string, variable scope

    :return normed: batch-normalized maps
    """
    mean, var = tf.nn.moments(x, axes, name='moments')

    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    with tf.name_scope('bn'):

        def mean_var_with_update():
            ema_apply_op = ema.apply([vars['mean'], vars['var']])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(vars['mean']), tf.identity(vars['var'])

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(vars['mean']), ema.average(vars['var'])))

        normed = tf.nn.batch_normalization(x, mean, var, vars['beta'], vars['gamma'], 1e-3)
    return normed

c0_size = 64
c1_size = 128
c2_size = 256
c3_size = 512
c34_size = 1024
c4_size = 1024

fc1_size = 2048
fc2_size = 256
out_size = 96
weight_stddev = 0.3
bias_stddev = 0.03

class RadNetModel(object):
    '''Implements the Radiation model for Climate Science

    TODO: Usage...

    '''

    def __init__(self):
        ''' Initializes the RadNet Model. '''
        self.vars = self._create_variables()
        self.phase_train = tf.placeholder(tf.bool, name="train_bool_node")


    def train_phase(self):
        return self.phase_train


    def _create_variables(self):
        var = dict()

        with tf.variable_scope('radnet'):
            with tf.variable_scope('conv0'):
                current = dict()
                current['w'] = weightInitilization5(1, 1, 1, c0_size, weight_stddev)
                current['b'] = biasInitialization(c0_size, bias_stddev)
                current['bn'] = bnInitialization(c0_size)
                current['pr'] = preluInitialization(c0_size)
                var['conv0'] = current
            with tf.variable_scope('conv00'):
                current = dict()
                current['w'] = weightInitilization5(3, 3, c0_size, c0_size, weight_stddev)
                current['b'] = biasInitialization(c0_size, bias_stddev)
                current['bn'] = bnInitialization(c0_size)
                current['pr'] = preluInitialization(c0_size)
                var['conv00'] = current
            with tf.variable_scope('conv1'):
                current = dict()
                current['w'] = weightInitilization5(3, 3, c0_size, c1_size, weight_stddev)
                current['b'] = biasInitialization(c1_size, bias_stddev)
                current['bn'] = bnInitialization(c1_size)
                current['pr'] = preluInitialization(c1_size)
                var['conv1'] = current
            with tf.variable_scope('conv11'):
                current = dict()
                current['w'] = weightInitilization5(3, 3, c1_size, c1_size, weight_stddev)
                current['b'] = biasInitialization(c1_size, bias_stddev)
                current['bn'] = bnInitialization(c1_size)
                current['pr'] = preluInitialization(c1_size)
                var['conv11'] = current
            with tf.variable_scope('conv2'):
                current = dict()
                current['w'] = weightInitilization5(3, 3, c1_size, c2_size, weight_stddev)
                current['b'] = biasInitialization(c2_size, bias_stddev)
                current['bn'] = bnInitialization(c2_size)
                current['pr'] = preluInitialization(c2_size)
                var['conv2'] = current
            with tf.variable_scope('conv22'):
                current = dict()
                current['w'] = weightInitilization5(3, 3, c2_size, c2_size, weight_stddev)
                current['b'] = biasInitialization(c2_size, bias_stddev)
                current['bn'] = bnInitialization(c2_size)
                current['pr'] = preluInitialization(c2_size)
                var['conv22'] = current
            with tf.variable_scope('conv3'):
                current = dict()
                current['w'] = weightInitilization5(3, 3, c2_size, c3_size, weight_stddev)
                current['b'] = biasInitialization(c3_size, bias_stddev)
                current['bn'] = bnInitialization(c3_size)
                current['pr'] = preluInitialization(c3_size)
                var['conv3'] = current
            with tf.variable_scope('conv33'):
                current = dict()
                current['w'] = weightInitilization5(3, 3, c3_size, c3_size, weight_stddev)
                current['b'] = biasInitialization(c3_size, bias_stddev)
                current['bn'] = bnInitialization(c3_size)
                current['pr'] = preluInitialization(c3_size)
                var['conv33'] = current

            with tf.variable_scope('conv4'):
                current = dict()
                current['w'] = weightInitilization5(3, 3, c3_size, c3_size, weight_stddev)
                current['b'] = biasInitialization(c3_size, bias_stddev)
                current['bn'] = bnInitialization(c3_size)
                current['pr'] = preluInitialization(c3_size)
                var['conv4'] = current

            with tf.variable_scope('conv44'):
                current = dict()
                current['w'] = weightInitilization5(3, 3, c3_size, c3_size, weight_stddev)
                current['b'] = biasInitialization(c3_size, bias_stddev)
                current['bn'] = bnInitialization(c3_size)
                current['pr'] = preluInitialization(c3_size)
                var['conv44'] = current



            with tf.variable_scope('fc1'):
                current = dict()

                current['w'] = weightInitilization3(2 * 2 * c3_size, fc1_size, weight_stddev)
                current['b'] = biasInitialization(fc1_size, bias_stddev)
                current['bn'] = bnInitialization(fc1_size)
                current['pr'] = preluInitialization(fc1_size)
                var['fc1'] = current
            with tf.variable_scope('fc2'):
                current = dict()
                current['w'] = weightInitilization3(fc1_size, fc2_size, weight_stddev)
                current['b'] = biasInitialization(fc2_size, bias_stddev)
                current['bn'] = bnInitialization(fc2_size)
                current['pr'] = preluInitialization(fc2_size)
                var['fc2'] = current
            with tf.variable_scope('out'):
                current = dict()
                current['w'] = weightInitilization3(fc2_size, out_size, weight_stddev)
                current['b'] = biasInitialization(out_size, bias_stddev)
                var['out'] = current

        return var

    def _create_network(self, input_batch):
        ''' Construct the network.'''

        print(input_batch.get_shape())
        # Pre-process the input
        # x is 64 x 1 tensor with padding at the end
        input_batch = tf.reshape(input_batch, shape=[-1, 196], name="input_node")
        input_batch = tf.reshape(input_batch, shape=[-1, 14, 14, 1], name="input_node_reshaped")


        with tf.name_scope('conv0'):
            #1x1 conv layer https://www.quora.com/What-is-a-1X1-convolution
            conv0 = conv2d(input_batch, self.vars['conv0']['w'], self.vars['conv0']['b'], strides=1, padding="VALID")
            conv0 = ReLU(conv0, self.vars['conv0']['pr'])
            conv0 = batchNorm(conv0, [0, 1, 2], self.vars['conv0']['bn'], self.phase_train)
            #conv0 = pool2d(conv0, k=1)
            print(conv0.get_shape())
        with tf.name_scope('conv00'):
            #1x1 conv layer https://www.quora.com/What-is-a-1X1-convolution
            conv0 = conv2d(conv0, self.vars['conv00']['w'], self.vars['conv00']['b'], strides=1)
            conv0 = ReLU(conv0, self.vars['conv00']['pr'])
            conv0 = batchNorm(conv0, [0, 1, 2], self.vars['conv00']['bn'], self.phase_train)
            #conv0 = pool2d(conv0, k=1)
            print(conv0.get_shape())
        with tf.name_scope('conv1'):
            conv1 = conv2d(conv0, self.vars['conv1']['w'], self.vars['conv1']['b'], strides=1)
            conv1 = batchNorm(conv1, [0, 1, 2], self.vars['conv1']['bn'], self.phase_train)
            print(conv1.get_shape())
            #conv1 = pool2d(conv1, k=1, l=1)
            conv1 = ReLU(conv1, self.vars['conv1']['pr'])
            print(conv1.get_shape())
        with tf.name_scope('conv11'):
            conv1 = conv2d(conv1, self.vars['conv11']['w'], self.vars['conv11']['b'], strides=1)
            conv1 = batchNorm(conv1, [0, 1, 2], self.vars['conv11']['bn'], self.phase_train)
            print(conv1.get_shape())
            conv1 = pool2d(conv1, k=1, l=1)
            conv1 = ReLU(conv1, self.vars['conv11']['pr'])
            print(conv1.get_shape())
        with tf.name_scope('conv2'):
            conv2 = conv2d(conv1, self.vars['conv2']['w'], self.vars['conv2']['b'], strides=1)
            print(conv2.get_shape())
            #conv2 = pool2d(conv2, k=1, l=1)
            conv2 = ReLU(conv2, self.vars['conv2']['pr'])
            conv2 = batchNorm(conv2, [0, 1, 2], self.vars['conv2']['bn'], self.phase_train)
            print(conv2.get_shape())
        with tf.name_scope('conv22'):
            conv2 = conv2d(conv2, self.vars['conv22']['w'], self.vars['conv22']['b'], strides=1)
            print(conv2.get_shape())
            conv2 = pool2d(conv2, k=2, l=2)
            conv2 = ReLU(conv2, self.vars['conv22']['pr'])
            conv2 = batchNorm(conv2, [0, 1, 2], self.vars['conv22']['bn'], self.phase_train)
            print(conv2.get_shape())
        with tf.name_scope('conv3'):
            conv3 = conv2d(conv2, self.vars['conv3']['w'], self.vars['conv3']['b'], strides=1)
            #conv3 = pool2d(conv3, k=1, l=1)
            conv3 = ReLU(conv3, self.vars['conv3']['pr'])
            conv3 = batchNorm(conv3, [0, 1, 2], self.vars['conv3']['bn'], self.phase_train)
            print(conv3.get_shape())
        with tf.name_scope('conv33'):
            conv3 = conv2d(conv3, self.vars['conv33']['w'], self.vars['conv33']['b'], strides=1)
            conv3 = pool2d(conv3, k=2, l=2)
            conv3 = ReLU(conv3, self.vars['conv33']['pr'])
            conv3 = batchNorm(conv3, [0, 1, 2], self.vars['conv33']['bn'], self.phase_train)
            print(conv3.get_shape())
            
        with tf.name_scope('conv4'):
            conv4 = conv2d(conv3, self.vars['conv4']['w'], self.vars['conv4']['b'], strides=1)
            #conv4 = pool2d(conv4, k=1, l=1)
            conv4 = ReLU(conv4, self.vars['conv4']['pr'])
            conv4 = batchNorm(conv4, [0, 1, 2], self.vars['conv4']['bn'], self.phase_train)
            print(conv4.get_shape())
        with tf.name_scope('conv44'):
            conv4 = conv2d(conv4, self.vars['conv44']['w'], self.vars['conv44']['b'], strides=1)
            conv4 = pool2d(conv4, k=2, l=2)
            conv4 = ReLU(conv4, self.vars['conv44']['pr'])
            conv4 = batchNorm(conv4, [0, 1, 2], self.vars['conv44']['bn'], self.phase_train)
            print(conv4.get_shape())
        """
        with tf.name_scope('conv4'):
            conv4 = conv2d(conv3, self.vars['conv4']['w'], self.vars['conv4']['b'], strides=1)
            print(conv4.get_shape())
            conv4 = pool2d(conv4, k=1, l=1)
            conv4 = ReLU(conv4, self.vars['conv4']['pr'])
            conv4 = batchNorm(conv4, [0, 1, 2], self.vars['conv4']['bn'], self.phase_train)

            print(conv4.get_shape())

        with tf.name_scope('conv5'):
            conv5 = conv2d(conv4, self.vars['conv5']['w'], self.vars['conv5']['b'], strides=1)
            print(conv5.get_shape())
            conv5 = pool2d(conv5, k=2, l=2)
            conv5 = ReLU(conv5, self.vars['conv5']['pr'])
            conv5 = batchNorm(conv5, [0, 1, 2], self.vars['conv5']['bn'], self.phase_train)

            print(conv5.get_shape())
        """

        with tf.name_scope('fc1'):
            # Reshape conv3 output to fit fully connected layer input
            fc1 = tf.reshape(conv4, [-1, self.vars['fc1']['w'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.vars['fc1']['w']), self.vars['fc1']['b'])
            fc1 = ReLU(fc1, self.vars['fc1']['pr'])
            fc1 = batchNorm(fc1, [0], self.vars['fc1']['bn'], self.phase_train)

            #print(fc1.get_shape())
        with tf.name_scope('fc2'):
            fc2 = tf.add(tf.matmul(fc1, self.vars['fc2']['w']), self.vars['fc2']['b'])
            fc2 = ReLU(fc2, self.vars['fc2']['pr'])
            fc2 = batchNorm(fc2, [0], self.vars['fc2']['bn'], self.phase_train)

        with tf.name_scope('out'):
            out = tf.add(tf.matmul(fc2, self.vars['out']['w']), self.vars['out']['b'], name="output_node")
            print(out.get_shape())
        return out

    def loss(self, input_batch, real_output):
        ''' Creates a RadNet network and returns the autoencoding loss.
            The variables are all scoped to the given name.

            TODO: Implement an option to choose between mse loss and huber loss.
        '''
        with tf.name_scope('radnet'):
            output = self._create_network(input_batch)
            with tf.name_scope('loss'):
                # Huber loss
                loss = tf.reduce_mean(self.huber_loss(real_output, output))
                print(loss.get_shape())
                # MSE loss
                #los = tf.reduce_mean(tf.squared_difference(output, real_output))
                #tf.scalar_summary('loss', loss)
                #print(los.get_shape())
                tf.summary.scalar('loss', loss)

                return loss

    def predict(self, input, real_output, id_file):
        """ Function for calculating prediction without backpropagating the error

        :param input:
        :param real_output:
        :param id_file:
        :return:
        """

        with tf.name_scope('radnet'):
            pred_output = self._create_network(input)

            # the loss here is mse because this call is made online for testing, not for training
            mse = tf.reduce_mean(tf.squared_difference(pred_output, real_output))
            mape = tf.multiply(100.0, tf.reduce_mean(tf.abs(tf.divide((real_output - pred_output), real_output))))

            return id_file, real_output, pred_output, mse, mape, input

    def huber_loss(self, y_true, y_pred, max_grad=1.):
        """Calculates the huber loss.
        Ref.: http://stackoverflow.com/questions/39106732/how-do-i-combine-tf-absolute-and-tf-square-to-create-the-huber-loss-function-in

        Parameters
        ----------
        y_true: np.array, tf.Tensor
          Target value.
        y_pred: np.array, tf.Tensor
          Predicted value.
        max_grad: float, optional
          Positive floating point value. Represents the maximum possible
          gradient magnitude.

        Returns
        -------
        tf.Tensor
          The huber loss.
        """
        err = tf.abs(y_true - y_pred, name='abs')
        mg = tf.constant(max_grad, name='max_grad')
        lin = mg * (err - .5 * mg)
        quad = .5 * err * err
        return tf.where(err < mg, quad, lin)