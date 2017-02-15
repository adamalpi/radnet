import numpy as np
import os
import fnmatch

def find_files(directory, pattern='*.csv'):
    ''' Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_data_samples(directory):
    ''' Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    print("files length: {}".format(len(files)))

    for filename in files:
        with open(filename) as f:
            lines = f.readlines()
            data = []
            label = []
            for j in range(1, len(lines)):
                items = lines[j].strip().split(",")
                T=float(items[2])
                Q=float(items[3])
                R=float(items[4])
                yield T, Q, R

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




vars = OnlineStats()

oR = OnlineStats(ddof=0)
oQ = OnlineStats(ddof=0)
oT = OnlineStats(ddof=0)
dir = '/Users/adam13/Documents/uni/TFM/Data/radiation_data_v2/'

iterator = load_data_samples(dir)

i = 0
for t, q, r in iterator:
    i += 1
    oT.include(t)
    oQ.include(q)
    oR.include(r)

    print(i)

print('T:  min {:.10f} max {:.10f} mean {:.10f} std {:.10f}'.format(oT.min, oT.max, oT.mean, oT.std))
print('Q:  min {:.10f} max {:.10f} mean {:.10f} std {:.10f}'.format(oQ.min, oQ.max, oQ.mean, oQ.std))
print('R:  min {:.10f} max {:.10f} mean {:.10f} std {:.10f}'.format(oR.min, oR.max, oR.mean, oR.std))





















