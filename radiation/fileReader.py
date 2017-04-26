import fnmatch
import os
import random
import re
import threading
import json

import tensorflow as tf


"""
Data v7: statistical values
ST:  min 100.0000000000 max 333.1499946801 mean 268.1406929063 std 37.5368706325
C:  min 0.0000000000 max 0.0099999904 mean 0.0017284657 std 0.0023850203
R:  min -255.9600440474 max 78.3198382662 mean -1.3758656541 std 6.1112494507
T:  min 100.0000000000 max 355.5721906214 mean 230.1788102309 Td 46.5063403685
H:  min -2720.3344538111 max 1848.3667831706 mean 4.2050377031 Hd 13.4852605066
"""

# data 7
minT = 100.0000000000
maxT = 355.5721906214
meanT = 230.1788102309
stdT = 46.5063403685

minH = -2720.3344538111
maxH = 1848.3667831706
meanH = 4.2050377031
stdH = 13.4852605066

minR = -255.9600440474
maxR = 78.3198382662
meanR = -1.3758656541
stdR = 6.1112494507

meanST = 268.1406929063
stdST = 37.5368706325

meanCO2 = 0.0017284657
stdCO2 = 0.0023850203

epoch = 0


def normalizeT(t):
    return normalize(t, minT, maxT, meanT, stdT)


def normalizeH(h):
    return normalize(h, minH, maxH, meanH, stdH)


def normalizeR(r):
    return normalize(r, minR, maxR, meanR, stdR)


def normalizeST(st):
    return normalize(st, 0, 0, meanST, stdST)


def normalizeCO2(c):
    return normalize(c, 0, 0, meanCO2, stdCO2)


def normalize(x, min, max, mean, std):
    # TODO: add option to choose between min-max of zero-mean normalization
    # return  (x - min) / (max - min) # min max normalization
    return (x - mean) / std  # standardization - zero-mean normalization
    # return x+100


def get_category_cardinality(files):
    """ Deprecated: function used before for identifying the samples based in its name
    and calculate the minimum and maximum sample id
    :param files: array of root paths.
    :return: min_id, max_id: int
    """
    file_pattern = r'([0-9]+)\.csv'
    id_reg_expression = re.compile(file_pattern)
    min_id = None
    max_id = None
    for filename in files:
        id = int(id_reg_expression.findall(filename)[0])
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    """ Function that randomizes a list of filePaths.

    :param files: list of path files
    :return: iterable of random files
    """
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.csv'):
    """ Recursively finds all files matching the pattern.

    :param directory:  directory path
    :param pattern: reggex
    :return: list of files
    """

    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_data_samples(files):
    """ Generator that yields samples from the directory.

    In the latest versions, the files are files where each line is a sample
    in json format. This function basically read each sample of the file and
    normalizes it and generates the data for the model in the desired format.

    :param files: list of files
    :return: iterable that contains the data, the label and the identifier of the sample.
    """

    for filename in files:
        with open(filename) as f:
            id = 0
            for line in f:
                id += 1
                input = json.loads(line)
                # todo normalize the input
                data = []
                label = []

                data.append((input['co2']))
                data.append((input['surface_temperature']))
                for i in range (0, len(input['radiation'])):
                    data.append(normalizeT(input['air_temperature'][i]))
                    data.append(normalizeH(input['humidity'][i]))
                    label.append((input['radiation'][i]))

                # fill last 2 values with 0
                for _ in range(0, 196-194):
                    data.append(0.0)

                yield data, label, [id]


class FileReader(object):
    """ Background reader that pre-processes radiation files
    and enqueues them into a TensorFlow queue.

    """

    def __init__(self,
                 data_dir,
                 coord,
                 n_input=196,
                 n_output=96,
                 queue_size=5000000,
                 test_percentage=0.2):

        # TODO: Implement a option that enables the usage of a test queue, by default it is
        # enabled here. For implementing this, the flag should be propagated to the several
        # functions that operate with both queues.

        self.data_dir = data_dir
        self.coord = coord
        self.n_input = n_input
        self.n_output = n_output
        self.threads = []
        self.sample_placeholder_train = tf.placeholder(tf.float32, [n_input])
        self.result_placeholder_train = tf.placeholder(tf.float32, [n_output])
        self.sample_placeholder_test = tf.placeholder(tf.float32, [n_input])
        self.result_placeholder_test = tf.placeholder(tf.float32, [n_output])
        self.idFile_placeholder_test = tf.placeholder(tf.int32, [1])
        self.idFile_placeholder_train = tf.placeholder(tf.int32, [1])

        self.queue_train = tf.PaddingFIFOQueue(queue_size, [tf.float32, tf.float32, tf.int32],
                                         shapes=[[n_input],[n_output], [1]])
        self.queue_test = tf.PaddingFIFOQueue(queue_size, [tf.float32, tf.float32, tf.int32],
                                              shapes=[[n_input], [n_output], [1]])
        self.enqueue_train = self.queue_train.enqueue([self.sample_placeholder_train, self.result_placeholder_train, self.idFile_placeholder_train])
        self.enqueue_test = self.queue_test.enqueue([self.sample_placeholder_test, self.result_placeholder_test, self.idFile_placeholder_test])

        # https://github.com/tensorflow/tensorflow/issues/2514
        # https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/rmGu1HAyPw4
        # Use of a flag that changes the input queue to another one, this way the model can
        # be tested using the test queue when required.
        self.select_q = tf.placeholder(tf.int32, [])
        self.queue = tf.QueueBase.from_list(self.select_q, [self.queue_train, self.queue_test])

        # Find any file as the reggex is *
        self.files = find_files(data_dir,'*')
        if not self.files:
            raise ValueError("No data files found in '{}'.".format(data_dir))

        print("files length: {}".format(len(self.files)))

        # Split the data into test and train datasets
        range = int(len(self.files) * (1-test_percentage))
        self.test_dataset = self.files[:range]
        self.train_dataset = self.files[range:]

    def dequeue(self, num_elements):
        """ Function for dequeueing a mini-batch

        :param num_elements: int size of minibatch
        :return:
        """
        data, label, id = self.queue.dequeue_many(num_elements)

        return data, label, id

    def queue_switch(self):
        return self.select_q

    def thread_main(self, sess, id, test):
        """ Thread function to be launched as many times as required for loading the data
        from several files into the Tensorflow's queue.

        :param sess: Tensorflow's session
        :param id: thread ID
        :param test: bool for choosing between the queue to feed the data, True for test queue
        :return: void
        """
        global epoch
        stop = False
        # Go through the dataset multiple times
        if test:
            files = self.test_dataset
        else:
            files = self.train_dataset

        # while tensorflows coordinator doesn't want to stop, continue.
        while not stop:

            epoch += 1
            if not test:
                print ("Number of epochs: {}".format(epoch))
            randomized_files = randomize_files(files)
            iterator = load_data_samples(randomized_files)

            for data, label, id_file in iterator:
                # update coordinator's state
                if self.coord.should_stop():
                    stop = True
                    break

                if test:  # in train range and test thread
                    sess.run(self.enqueue_test,
                             feed_dict={self.sample_placeholder_test: data,
                                        self.result_placeholder_test: label,
                                        self.idFile_placeholder_test: id_file})
                else:  # below the rage -> train
                    sess.run(self.enqueue_train,
                             feed_dict={self.sample_placeholder_train: data,
                                        self.result_placeholder_train: label,
                                        self.idFile_placeholder_train: id_file})

    def start_threads(self, sess, n_threads=2):
        """ Reader threads' launcher, uses the first thread for feeding into the test queue
        and the rest for feeding into the train queue.

        :param sess:
        :param n_threads:
        :return: void
        """
        for id in range(n_threads):
            if id == 0:
                thread = threading.Thread(target=self.thread_main, args=(sess, id, True))
            else:
                thread = threading.Thread(target=self.thread_main, args=(sess, id, False))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

        return self.threads


