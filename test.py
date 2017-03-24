from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import os
import tensorflow as tf
import time
from radiation import RadNet

SAMPLES = 100
TEMPERATURE = 1.0
LOGDIR = './logdir'
DATA_DIRECTORY = './data'
OUTPUT_DIRECTORY = './climate_results'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())



def write_pred_file(id_file, original, prediction, loss):
    results_dir = os.path.join(OUTPUT_DIRECTORY, STARTED_DATESTRING)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(results_dir+'/'+str(id_file)+'.csv', 'w') as file:
        file.write(str(loss) + '\n')
        for ori, pred in zip(original, prediction):
            file.write(str(ori)+','+str(pred)+'\n')


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='RadNet test script')

    parser.add_argument(
        '--graph_location',
        type=str,
        default=DATA_DIRECTORY,
        help='The file in which the frozen graph is serialized')

    arguments = parser.parse_args()

    return arguments


def main():
    args = get_arguments()

    graph_full_radnet_path = "/Users/adam13/Documents/uni/TFM/logdir/train/2017-03-15T11-34-44/graph-empty-radnet.pb"
    graph_full_radnet_path = args.graph_location

    data = [-2.2790625949523307, -1.4088417453717172, -2.2790625949523307, -1.4088417453717172, -2.005761932145158,
            -1.4056919679032809, -1.4131073729909267, -1.3807523415712635, -1.3668854394686523,
            -1.3827810793741122, -0.6564711315884887, -1.1718722991493322, -0.41851506738043326,
            -1.0063592710511373, -0.2325802298321742, -0.8268304011418217, -0.0812688788046175,
            -0.6449016300073649, 0.045103312071981885, -0.46836012345822076, 0.1544823561426056,
            -0.2970339367006489, 0.25023935788690876, -0.1341213308603885, 0.3357837423587971,
            -0.01864298573547887, 0.41318365102161103, 0.17004558785887405, 0.48381137815631947,
            0.3115885553072702, 0.5484692464372332, 0.4451993384970228, 0.6085955834723655, 0.5735408366499881,
            0.6647721357864366, 0.6967633350139107, 0.7171554826467985, 0.8135581119681424, 0.7666061155416369,
            0.9261934931387995, 0.8132119526890544, 1.0337369167314392, 0.857773331484518, 1.1390215039003402,
            0.8996138308765764, 1.2375890637701554, 0.939867751409681, 1.3344610741463308, 0.978204361395038,
            1.427165290324485, 1.014651456188852, 1.5151173015723922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0]
    label = [1.7993109016293565, 2.057413833260893, 0.7576212626431458, -0.5669940849327731, 1.124539786862435,
             -2.810264240574702, -1.6079031196620117, -1.4577900456567432, -1.3376846090694008, -1.1608615984231405,
             -0.9434977702562289, -0.7065192795368237, -0.44335735842027585, -0.4036012286877049,
             -0.2612927981054907, -0.12707185010661254, 0.010192448740591008, 0.15912854242554783,
             0.3237280476114565, 0.48620090718653314, 0.6431653476310148, 0.7725174052879127, 0.8820135539344693,
             0.9349478815423498, 0.943724753181962, 0.7997059696731448]

    '''
        [[1.81301022  2.0554502   0.82125396 - 0.52040219  1.10219002
        - 2.80707812 - 1.62169087 - 1.49699783 - 1.32370114 - 1.17672646
          - 0.9917931 - 0.7299549 - 0.42831016 - 0.42341071
          - 0.24065816 - 0.16656844  0.02209616  0.1923828
          0.31080574  0.46831447  0.68001342  0.78660715  0.90280712
          0.89587611 0.94437349  0.80615807]]'''


    id_file = [1200000]

    start_time = time.time()
    radnet = RadNet(graph_full_radnet_path)
    duration = time.time() - start_time
    print('{:10f} secs loading the model'.format(duration))

    # Code todo in __call__

    for i in range(100):
        start_time = time.time()
        prediction = radnet.predict(data)
        duration = time.time() - start_time
        print('{:10f} sec/prediction'.format(duration))

    print("my prediction")
    print(prediction)


if __name__ == '__main__':
    main()