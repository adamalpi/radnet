from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import os

import numpy as np
import tensorflow as tf

from radiation import RadNetModel, FileReader, optimizer_factory

SAMPLES = 100
TEMPERATURE = 1.0
LOGDIR = './logdir'
DATA_DIRECTORY = './data'
OUTPUT_DIRECTORY = './climate_results'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())



def write_pred_file(id_file, original, prediction, mse, mape, input):
    results_dir = os.path.join(OUTPUT_DIRECTORY, STARTED_DATESTRING)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(results_dir+'/'+str(id_file)+'.csv', 'w') as file:
        file.write(str(mse)+','+str(mape)+','+str(input['co2']) +','+ str(input['surface_temperature'])+',' + '\n')
        for ori, pred, t, h in zip(original, prediction, input['air_temperature'], input['humidity']):
            file.write(str(ori)+','+str(pred)+','+str(t)+','+str(h)+'\n')





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

    parser = argparse.ArgumentParser(description='RadNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many samples to predict')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--out_path',
        type=str,
        default=OUTPUT_DIRECTORY,
        help='Path to output the samples')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=DATA_DIRECTORY,
        help='The dir in which are located the samples to predict')

    arguments = parser.parse_args()

    return arguments


def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)


    # Create coordinator.
    coord = tf.train.Coordinator()
    # Load reader
    with tf.name_scope('create_inputs'):

        reader = FileReader(args.data_dir, coord)
        data, label, id_file = reader.dequeue(1)

    sess = tf.Session()

    with tf.name_scope('create_model'):
        net = RadNetModel()
        prediction = net.predict(data, label, id_file)


    init = tf.global_variables_initializer()
    #init = tf.initialize_all_variables()
    sess.run(init, {net.train_phase(): False})

    saver = tf.train.Saver(var_list=tf.trainable_variables())
    print(tf.trainable_variables())
    for v in tf.trainable_variables():
        print(v.name)

    try:
        ckpt = tf.train.get_checkpoint_state(args.checkpoint)
        print('Restoring model from {}'.format(args.checkpoint))
        saver.restore(sess, ckpt.model_checkpoint_path)

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess, n_threads=2)

    try:
        last_sample_timestamp = datetime.now()
        n = 0
        for step in range(args.samples):
            n=n+1
            # Run the RadNet to predict the next sample.
            id_file, real_output, pred_output, mse, mape, input = sess.run(prediction, {reader.queue_switch(): 0,  net.train_phase(): False})

            data_input = reader.decompose_data(input.tolist()[0])
            write_pred_file(str(id_file.tolist()[0][0])+str(n), real_output.tolist()[0], pred_output.tolist()[0], mse, mape, data_input)
            #print(type(real_output))

            # Show progress only once per second.
            current_sample_timestamp = datetime.now()
            time_since_print = current_sample_timestamp - last_sample_timestamp
            if time_since_print.total_seconds() > 1.:
                print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples),
                      end='\r')
                last_sample_timestamp = current_sample_timestamp

            # If we have partial writing, save the result so far.
            #if (args.wav_out_path and args.save_every and (step + 1) % args.save_every == 0):


        # Introduce a newline to clear the carriage return from the progress.
        print('test')


        # Save the result as a wav file.
        # if args.wav_out_path:

        print('Finished generating. The result can be viewed in TensorBoard.')

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()