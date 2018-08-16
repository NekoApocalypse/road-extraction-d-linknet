import os
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'ckpt_dir', '/model/archive_0710', 'path to checkpoint files')


def freeze_graph(model_dir, output_nodes):
    """Extract the sub graph defined by the output nodes and
    convert all variables into constant
    Args:
        model_dir: string, folder containing target checkpoint files
        output_nodes: list of string, name of output tensors
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError('Target directory \'{}\' does not exist.'.format(
            model_dir
        ))

    if not output_nodes:
        raise AssertionError('Output nodes not provided.')

    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    if not checkpoint_state:
        raise AssertionError('No valid checkpoints found in directory' +
                             ' \'{}\''.format(model_dir))
    input_checkpoint = checkpoint_state.model_checkpoint_path
    target_file = os.path.join(model_dir, 'frozen_model.pb')
    clear_devices = True

    with tf.Session(graph=tf.Graph()) as sess:
        # restore metagraph
        saver = tf.train.import_meta_graph(
            input_checkpoint + '.meta', clear_devices=clear_devices
        )
        # restore weights
        saver.restore(sess, input_checkpoint)

        # for node in tf.get_default_graph().as_graph_def().node:
        #   print(node.name)

        # convert variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_nodes
        )
        # save graph to file
        with tf.gfile.GFile(target_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('Write complete. {} ops in final graph'.format(
            len(output_graph_def.node)))

    return output_graph_def


def print_nodes(model_dir):
    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint_state.model_checkpoint_path
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(
            input_checkpoint + '.meta', clear_devices=True)
        [print(node.name) for node in tf.get_default_graph().as_graph_def().node]

    return None


if __name__ == '__main__':
    # model_dir = './model/archive_0710'
    model_dir = FLAGS.ckpt_dir
    output_nodes = [
        'metrics/Sigmoid',
        'metrics/Cast'
    ]
    freeze_graph(model_dir, output_nodes)
    # print_nodes(model_dir)
