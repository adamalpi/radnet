"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.saved_model import builder as smb
from tensorflow.python.framework import graph_util

from radiation import RadNetModel, FileReader, optimizer_factory


# Hyperparameters and other variables
BATCH_SIZE = 64
DATA_DIRECTORY = './data'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 2000
NUM_STEPS = 150000
LEARNING_RATE = 1e-3
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
MOMENTUM = 0.9


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='RadNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many climate files to process at once.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the data corpus.')
    parser.add_argument('--store_metadata', type=bool, default=False,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                                               'used by sgd or rmsprop optimizer. Ignored by the '
                                               'adam optimizer.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option.')

    return parser.parse_args()



def save(saver, sess, logdir, step):
    """Saves Model"""
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')

def load(saver, sess, logdir):
    """Loads Model"""
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    print('test')
    print(ckpt)

    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root, dir='train'):
    logdir = os.path.join(logdir_root, dir, STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    logdir_root = directories['logdir_root']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    # Create coordinator.
    coord = tf.train.Coordinator()

    # Load data from data corpus.
    with tf.name_scope('create_inputs'):

        reader = FileReader(args.data_dir, coord)
        data, label, _ = reader.dequeue(args.batch_size)

    # Create network.
    with tf.name_scope('create_model'):
        net = RadNetModel()
        loss = net.loss(data, label)

    optimizer = optimizer_factory[args.optimizer](
                    learning_rate=args.learning_rate,
                    momentum=args.momentum)

    trainable = tf.trainable_variables() #todo: not sure if this works

    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up logging for TensorBoard.
    writer_train = tf.summary.FileWriter(logdir)
    writer_test = tf.summary.FileWriter(get_default_logdir(LOGDIR_ROOT, 'test'))
    #writer_train = tf.train.SummaryWriter(logdir)
    #writer_test = tf.train.SummaryWriter(get_default_logdir(LOGDIR_ROOT, 'test'))

    writer_train.add_graph(tf.get_default_graph())
    writer_test.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    # train_summary = tf.scalar_summary("train_loss", loss)
    # test_summary = tf.scalar_summary("test_loss", loss)

    # Set up session
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    init = tf.global_variables_initializer()
    #init = tf.initialize_all_variables()
    sess.run(init, {net.train_phase(): False})

    # Saver for storing checkpoints of the model.
    #saver = tf.train.Saver(var_list=tf.trainable_variables())
    saver = tf.train.Saver()

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    step = None
    try:
        last_saved_step = saved_global_step
        for step in range(saved_global_step + 1, args.num_steps):
            start_time = time.time()

            if args.store_metadata and step % 50 == 0:
                # Slow run that stores extra information for debugging.
                print('Storing metadata')
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                summary, loss_value, _ = sess.run(
                    [summaries, loss, optim],
                    options=run_options,
                    run_metadata=run_metadata)
                writer_train.add_summary(summary, step)
                writer_train.add_run_metadata(run_metadata,
                                        'step_{:04d}'.format(step))
                tl = timeline.Timeline(run_metadata.step_stats)
                timeline_path = os.path.join(logdir, 'timeline.trace')
                with open(timeline_path, 'w') as f:
                    f.write(tl.generate_chrome_trace_format(show_memory=True))
            else:
                summary, loss_value_train, _ = sess.run([summaries, loss, optim],
                                                        {reader.queue_switch(): 0, net.train_phase(): True})
                writer_train.add_summary(summary, step)




            if step % 100 == 0:
                summary, loss_value_test = sess.run([summaries, loss],
                                                    {reader.queue_switch(): 1, net.train_phase(): False})
                writer_test.add_summary(summary, step)

                duration = time.time() - start_time
                print('train: step {:d} - loss = {:.5f}, ({:.3f} sec/step)'
                      .format(step, loss_value_train, duration))
                print('test: step {:d} - loss = {:.5f}, ({:.3f} sec/step)'
                      .format(step, loss_value_test, duration))

            if step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)


        name_empty_graph_file = 'graph-empty-{}.pb'.format("radnet")
        name_full_graph_file = 'graph-full-{}.pb'.format("radnet")

        # Store empty graph file
        print('Saving const graph def to {}'.format(name_empty_graph_file))
        #graph_def = sess.graph_def
        #graph_def = sess.graph.as_graph_def()
        graph_def = tf.get_default_graph().as_graph_def()

        # fix batch norm nodes
        # https://github.com/tensorflow/tensorflow/issues/3628
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        converted_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ["create_model/radnet_1/out/output_node"])

        tf.train.write_graph(converted_graph_def, logdir, name_empty_graph_file, as_text=False)


        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
