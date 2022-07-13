import os
import tensorflow as tf

model_name = 'glasses'

model = tf.keras.models.load_model(f'models/{model_name}/keras/keras_model.h5')
if not os.path.exists(f'models/{model_name}/converted_savedmodel'):
    tf.saved_model.save(model, f'models/{model_name}')
    print(f'model converted: models/{model_name}')
else:
    print(f'converted model already exists: models/{model_name}')
