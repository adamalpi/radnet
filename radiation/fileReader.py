import fnmatch
import os
import random
import re
import threading
import json

import numpy as np
import tensorflow as tf


FILE_PATTERN = r'([0-9]+)\.csv'



"""
minT = -80.0000000000
maxT = 29.9879614723
meanT = -3.8945928534
stdT = 33.3932939425

minQ = 0.0000000010
maxQ = 27.2849571304
meanQ = 13.1466227845
stdQ = 9.3315113828

minR = -10.7046261119
maxR = 12.6840617592
meanR = -2.0750710241
stdR = 1.2158839662

R:  min -72.2602017533 max 20.1615070290 mean -0.6369230161 std 2.0326646987
T:  min 100.0000000000 max 333.1499604121 mean 268.1425003086 std 37.5403768822
C:  min 0.0000000000 max 0.0099999956 mean 0.0017302686 std 0.0023876463
"""
"""
minT = 100.0000000000
maxT = 355.8483002305
meanT = 230.1844755700
stdT = 46.5110756771

minH = -23.0441724939
maxH = 1.5476185910
meanH = 0.0042088778
stdH = 0.0136187270

minR = -72.2602017533
maxR = 20.1615070290
meanR = -0.6369230161
stdR = 2.0326646987

meanST = 268.1425003086
stdST = 37.5403768822

meanCO2 = 0.0017302686
stdCO2 = 0.0023876463

ST:  min 100.0000000000 max 300.0000000000 mean 300.0000000000 std 0.0000000000
C:  min 0.0003000000 max 0.0003000000 mean 0.0003000000 std 0.0000000000
R:  min -119.2513102444 max 39.7685035911 mean -2.2027801662 std 6.8570745796
T:  min 100.0000000000 max 328.1996104878 mean 262.0392112351 Td 27.4567452039
H:  min 0.0000000218 max 101.0662332648 mean 4.2796499939 Hd 6.1626364326
"""
"""
data_v4
minT = 100.0000000000
maxT = 328.1996104878
meanT = 262.0392112351
stdT = 27.4567452039

minH = 0.0000000218
maxH = 101.0662332648
meanH = 4.2796499939
stdH = 6.1626364326

minR = -119.2513102444
maxR = 39.7685035911
meanR = -2.2027801662
stdR = 6.8570745796

meanST = 300.0000000000
stdST = 0.0

meanCO2 = 0.0003000000
stdCO2 = 0.0

ST:  min 100.0000000000 max 300.0000000000 mean 300.0000000000 std 0.0000000000
C:  min 0.0003000000 max 0.0003000000 mean 0.0003000000 std 0.0000000000
R:  min -43.3588707167 max 7.0534840711 mean -2.2987180030 std 3.4057274863
T:  min 100.0000000000 max 327.2450548496 mean 260.3624504585 Td 27.1639526147
H:  min 0.0000001027 max 81.4849408735 mean 3.9085964279 Hd 5.6914521830
"""

"""
data_v5
minT = 100.0000000000
maxT = 327.2450548496
meanT = 260.3624504585
stdT = 27.1639526147

minH = 0.0000001027
maxH = 81.4849408735
meanH = 3.9085964279
stdH = 5.6914521830

minR = -43.3588707167
maxR = 7.0534840711
meanR = -2.2987180030
stdR = 3.4057274863

meanST = 300.0000000000
stdST = 0.0

meanCO2 = 0.0003000000
stdCO2 = 0.0

ST:  min 100.0000000000 max 333.1499923913 mean 268.1782537513 std 37.5253217546
C:  min 0.0000000001 max 0.0099999913 mean 0.0017316786 std 0.0023871746
R:  min -60.0309877358 max 13.4218149025 mean -1.4530629607 std 3.3745299288
T:  min 100.0000000000 max 355.3580271402 mean 228.5427318133 Td 46.3281757901
H:  min 0.0000000000 max 1246.3252393693 mean 3.9482684404 Hd 12.8864789784
"""
"""
data_v6
minT = 100.0000000000
maxT = 355.3580271402
meanT = 228.5427318133
stdT = 46.3281757901

minH = 0.0000001027
maxH = 1246.3252393693
meanH = 3.9482684404
stdH = 12.8864789784

minR = -43.3588707167
maxR = 7.0534840711
meanR = -1.4530629607
stdR = 3.3745299288

meanST = 268.1782537513
stdST = 37.5253217546

meanCO2 = 0.0017316786
stdCO2 = 0.0023871746

ST:  min 100.0000000000 max 333.1499946801 mean 268.1406929063 std 37.5368706325
C:  min 0.0000000000 max 0.0099999904 mean 0.0017284657 std 0.0023850203
R:  min -255.9600440474 max 78.3198382662 mean -1.3758656541 std 6.1112494507
T:  min 100.0000000000 max 355.5721906214 mean 230.1788102309 Td 46.5063403685
H:  min -2720.3344538111 max 1848.3667831706 mean 4.2050377031 Hd 13.4852605066
"""
"""
data 7
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

ST:  min 100.0000000000 max 300.0000000000 mean 300.0000000000 std 0.0000000000
C:  min 0.0003000000 max 0.0003000000 mean 0.0003000000 std 0.0000000000
R:  min -124.5939367995 max 41.6578092115 mean -2.2024066011 std 6.8566207055
T:  min 100.0000000000 max 330.6810650081 mean 262.0361042300 Td 27.4560401312
H:  min 0.0000000011 max 120.0698121190 mean 4.2776340796 Hd 6.1596216695
"""

minT = 100.0000000000
maxT = 330.6810650081
meanT = 262.0361042300
stdT = 27.4560401312

minH = 0.0000000011
maxH = 120.0698121190
meanH = 4.2776340796
stdH = 6.1596216695

minR = -124.5939367995
maxR = 41.6578092115
meanR = -2.2024066011
stdR = 6.8566207055

meanST = 300.0000000000
stdST = 0.0000000000

meanCO2 = 0.0003000000
stdCO2 = 0.0000000000

epoch = 0


def normalizeT(t):
    return normalize(t, minT, maxT, meanT, stdT)


def normalizeH(h):
    return normalize(h, minH, maxH, meanH, stdH)


def normalizeR(r):
    return normalize(r, minR, maxR, meanR, stdR)

def normalizeST(r):
    return normalize(r, minR, maxR, meanST, stdST)

def normalizeCO2(r):
    return normalize(r, minR, maxR, meanCO2, stdCO2)


def normalize(x, min, max, mean, std):
    # return  (x - min) / (max - min) # min max normalization
    return (x - mean) / std  # standardization - zero-mean normalization
    # return x+100


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
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
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.csv'):
    ''' Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_data_samples(files):
    ''' Generator that yields samples from the directory.'''
    #files = find_files(directory, '*')
    #print("files length: {}".format(len(files)))
    # id_reg_expression = re.compile(FILE_PATTERN)
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

                #for i in range (0, len(input['radiation'])):


                #for i in range (0, len(input['radiation'])):
                #    data.append(normalizeCO2(input['co2']))

                #for i in range (0, len(input['radiation'])):
                #    data.append(normalizeST(input['surface_temperature']))

                #fill last 2 values with 0
                for _ in range(0, 196-194):
                    data.append(0.0)

                yield data, label, [id]




class FileReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 data_dir,
                 coord,
                 n_input=196,
                 n_output=96,
                 queue_size=5000000,
                 test_percentage=0.2):

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
        # https: // groups.google.com / a / tensorflow.org / forum /  # !topic/discuss/rmGu1HAyPw4
        self.select_q = tf.placeholder(tf.int32, [])
        self.queue = tf.QueueBase.from_list(self.select_q, [self.queue_train, self.queue_test])

        self.files = find_files(data_dir,'*')
        if not self.files:
            raise ValueError("No data files found in '{}'.".format(data_dir))

        print("files length: {}".format(len(self.files)))

        range = int(len(self.files) * (1-test_percentage))
        self.test_dataset = self.files[:range]
        self.train_dataset = self.files[range:]

        #min_id, max_id = get_category_cardinality(self.files)
        #self.test_range = max_id-(max_id-min_id)*test_percentage

    def dequeue(self, num_elements):
        data, label, id = self.queue.dequeue_many(num_elements)

        return data, label, id

    def queue_switch(self):
        return self.select_q

    def thread_main(self, sess, id, test):
        global epoch
        stop = False
        # Go through the dataset multiple times
        if test:
            files = self.test_dataset
        else:
            files = self.train_dataset


        while not stop:
            epoch += 1
            if not test:
                print ("Number of epochs: {}".format(epoch))
            randomized_files = randomize_files(files)
            iterator = load_data_samples(randomized_files)

            for data, label, id_file in iterator:
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

        for id in range(n_threads):
            if id == 0:
                thread = threading.Thread(target=self.thread_main, args=(sess, id, True))
            else:
                thread = threading.Thread(target=self.thread_main, args=(sess, id, False))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads


# Script for determining the max, min, mean and std