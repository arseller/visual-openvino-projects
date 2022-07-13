import tensorflow as tf

model_name = 'glasses'

model = tf.keras.models.load_model('models/keras_model.h5')
tf.saved_model.save(model, 'model')
