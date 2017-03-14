from tensorflow.python.tools import freeze_graph
import tensorflow as tf

'''
from tensorflow.python.radiation import megafunction
rad = megafunction.Radiation()
rad.infer(sample[])
'''



import sys, os

def main():
    args = sys.argv
    logdir = args[1]
    name_empty_graph_file = 'graph-empty-{}.pb'.format("radnet")
    name_full_graph_file = 'graph-full-{}.pb'.format("radnet")


    #Merge the graph file together with the variables of the last checkpoint into a single
    #file that will be used for building a inferenciating application
    checkpoint_state_name = "checkpoint_state"


    input_graph_path = os.path.join(logdir, name_empty_graph_file)
    input_saver_def_path = ""

    ckpt = tf.train.get_checkpoint_state(logdir)
    print ("checkpoint file: {}".format(ckpt.model_checkpoint_path))
    input_checkpoint_path = ckpt.model_checkpoint_path

    # Note that we this normally should be only "output_node"!!!
    output_node_names = "create_model/radnet_1/out/output_node"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(logdir, name_full_graph_file)
    clear_devices = False
    initializer_nodes = False
    input_binary = True

    freeze_graph.freeze_graph(input_graph_path,
                              input_saver_def_path,
                              input_binary,
                              input_checkpoint_path,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              output_graph_path,
                              clear_devices,
                              initializer_nodes)


if __name__ == '__main__':
    main()