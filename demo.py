from keras import models, layers
import numpy as np
import sincnet

# Create the model.
model = models.Sequential()
#model.add(sincnet.SincConv1D(80, 251, 16000, input_shape=(1024, 1))) # TODO is the shape good?
model.add(sincnet.SincConv1D(2, 25, 16000, input_shape=(1024, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(10))
model.summary()

# Do a prediction.
prediction = model.predict(np.random.random((1, 1024, 1)))
print(prediction.shape)
