import os
import tensorflow as tf

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

folder_name = 'nofuzzy_model'
# model path
frozen_out_path = f'models/{folder_name}/frozen_graph'
if not os.path.exists(frozen_out_path):
    os.makedirs(frozen_out_path)
    print(f'{frozen_out_path} created')
# name of the .pb file
frozen_graph_filename = f'models/{folder_name}/frozen_graph'

if not os.path.exists(f'{frozen_graph_filename}/frozen_graph.pb'):

    model = tf.keras.models.load_model(f'models/{folder_name}/converted_savedmodel/model.savedmodel')

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen graph def
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 60)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 60)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print()
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    print()

    try:
        # Save frozen graph to disk
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=frozen_out_path,
                          name='frozen_graph.pb',
                          as_text=False)
        # Save its text representation
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=frozen_out_path,
                          name='frozen_graph.pbtxt',
                          as_text=True)

    finally:
        print('\nmodel converted: ', f'{frozen_graph_filename}/frozen_graph.pb')

else:
    print('\nmodel already exists: ', f'{frozen_graph_filename}/frozen_graph.pb')
