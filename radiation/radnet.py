# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc#.dykqbzqek
# https://medium.com/@hamedmp/exporting-trained-tensorflow-models-to-c-the-right-way-cf24b609d183#.2yffldwf7
# https://github.com/tensorflow/tensorflow/issues/616


from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops


class RadNet:
    """ Class for loading and fetching the model for getting predictions.

    """

    def __init__(self, frozen_graph_path):
        """ Initializes the needed variables for making prediction

        Loads the graph from a frozen graph in protobbuf format file generated after training the model.
        Then starts a Tensorflow session with that graph and gets the needed variables for
        fetching the model in further calls. The session is then initialized once.

        :param frozen_graph_path: path to the protobuf (.pb) file containing the graph and the value
        of the variables.
        """

        with ops.Graph().as_default():
            output_graph_def = graph_pb2.GraphDef()

            with open(frozen_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = importer.import_graph_def(output_graph_def, name="")



            # We are initializing the session with the default graph which already has the
            # frozen graph loaded.
            self.sess = session.Session()

            for op in self.sess.graph.get_operations():
                print(op.name)

            # The input and output nodes are gotten into a variable
            self.input_node = self.sess.graph.get_tensor_by_name("create_model/radnet_1/input_node:0")
            self.output_node = self.sess.graph.get_tensor_by_name("create_model/radnet_1/out/output_node:0")
            # Getting the train_flag_node is also required as it is like that and we need to indicate the graph
            # we are just fetching a parameter.
            self.train_flag_node = self.sess.graph.get_tensor_by_name("create_model/train_bool_node:0")

    def predict(self, sample):
        """ Method for fetching the model

        :param sample: array[64] with the correct input of the model
        :return prediction: array[26] with the radiation level for each of the 26 layers
        """

        prediction = self.sess.run(
                self.output_node,
                feed_dict={
                     self.input_node: [sample],
                     self.train_flag_node: False
                })

        return prediction
