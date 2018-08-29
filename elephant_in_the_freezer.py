import os
import glob
import tensorflow as tf


def load_graph(pb_dir):
    pb_file = glob.glob(
        os.path.join(pb_dir, '*.pb')
    )[0]
    with tf.gfile.GFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="model")

    # Input: 'prefix/input_x'
    # Binary Prediction: 'prefix/metrics/Sigmoid'
    # Float Prediction: 'prefix/metrics/Cast'
    names = [op.name for op in graph.get_operations()]
    [print(name) for name in names]
    x = graph.get_tensor_by_name('model/input_x:0')
    pred = graph.get_tensor_by_name('model/metrics/Sigmoid:0')
    bin_pred = graph.get_tensor_by_name('model/metrics/Cast:0')
    endpoints = {
        'x': x,
        'pred': pred,
        'bin_pred': bin_pred
    }
    # print(x, pred, bin_pred)
    return graph, endpoints


if __name__ == '__main__':
    pb_dir = './model/dt_0815_resume'
    graph, endpoints = load_graph(pb_dir)
    [print(endpoint) for endpoint in endpoints.items()]
    print(endpoints['x'])
