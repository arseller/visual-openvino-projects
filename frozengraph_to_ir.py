import os

input_model_name = 'detection_warning'

input_model_path = f'models/{input_model_name}/frozen_graph/frozen_graph.pb'
input_shape = '[1,224,224,3]'
mean = '[127.5,127.5,127.5]'
scale = '[127.0]'
precision = 'FP16'
output_model_path = f'models/{input_model_name}/ir_model'
if not os.path.exists(output_model_path):
    os.makedirs(output_model_path)
    print(f'{output_model_path} created')
output_model_name = f'{output_model_path}/frozen_graph.xml'

if not os.path.exists(output_model_name):
    try:
        convert_cmd = f'mo ' \
                      f'--input_model "{input_model_path}" ' \
                      f'--input_shape "{input_shape}" ' \
                      f'--mean_value "{mean}" ' \
                      f'--scale_value "{scale}" ' \
                      f'--data_type {precision} ' \
                      f'--output_dir "{output_model_path}"'

        os.system(convert_cmd)

    finally:
        print('\nmodel converted: ', output_model_name)

else:
    print('\nmodel already exists: ', output_model_name)

