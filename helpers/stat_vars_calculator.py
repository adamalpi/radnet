import numpy as np
import os
import json
import fnmatch

def find_files(directory, pattern='*.csv'):
    ''' Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_data_samples(directory):
    ''' Generator that yields samples from the directory.'''
    files = find_files(directory, '*')
    print(files)
    print("files length: {}".format(len(files)))
    # id_reg_expression = re.compile(FILE_PATTERN)
    for filename in files:
        print(filename)
        with open(filename) as f:
            id = 0
            for line in f:
                id += 1
                input = json.loads(line)
                # todo normalize the input
                data = []
                data.append(input['co2'])
                data.append(input['surface_temperature'])
                for i in range (0, len(input['radiation'])):
                #    data.append(input['humidity'][i])
                #    data.append(input['air_temperature'][i])
                    yield input['radiation'][i]*86400

#http://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
class OnlineStats(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        self.min, self.max = 100, -100
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        if self.min>datum:
            self.min = datum
        if self.max<datum:
            self.max = datum

        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)
        self.variance = self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)




dir = '/home/adam/data/data_v3/'

iterator = load_data_samples(dir)

vars = OnlineStats()

oR = OnlineStats(ddof=0)
oQ = OnlineStats(ddof=0)
oT = OnlineStats(ddof=0)

i = 0
for r in iterator:
    i += 1
    #oT.include(t)
    #oQ.include(q)
    oR.include(r)
    if (i%100000 == 0):
        print(i)

#print('T:  min {:.10f} max {:.10f} mean {:.10f} std {:.10f}'.format(oT.min, oT.max, oT.mean, oT.std))
#print('Q:  min {:.10f} max {:.10f} mean {:.10f} std {:.10f}'.format(oQ.min, oQ.max, oQ.mean, oQ.std))
print('R:  min {:.10f} max {:.10f} mean {:.10f} std {:.10f}'.format(oR.min, oR.max, oR.mean, oR.std))






















