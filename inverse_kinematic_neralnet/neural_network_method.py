# %%
# Import Lib
import numpy as np
import tensorflow as tf                        #tensorflow achitecture
from tensorflow import keras                   #Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.
import matplotlib.pyplot as plt                #graph plot

# %%
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

# %%
print(x_train[0].shape)
print(y_train[0].shape)
print(x_test.shape)
print(y_test.shape)

# %%
y_train

# %%
# Difine Parameter
l1 = 2
l2 = 2
l3 = 2

q1 = np.linspace(0, np.pi, num=1000)
q2 = np.linspace(-np.pi, 0, num=1000)
q3 = np.linspace(-np.pi/2, np.pi/2, num=1000)

y_train = np.array([[q1[0],q2[0],q3[0]]])

for i in range(999):
  y_train = np.append(y_train,[[q1[i],q2[i],q1[i]]],axis=0)

# y_train = np.array([q1[0],q2[0],q3[0]])
# for i in range(999):
#   y_train = np.append(y_train,[q1[i],q2[i],q1[i]],axis=0)

y_train.shape

# %%
# Equation
def arm(q1,q2,q3):
  xE = l1*np.cos(q1) + l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3)
  yE = l1*np.sin(q1) + l2*np.sin(q1+q2) + l3*np.sin(q1+q2+q3)
  tE = q1+q2+q3
  return xE,yE,tE

xl,yl,tl = arm(q1,q2,q3)

x_train = np.array([[xl[0],yl[0],tl[0]]])

for i in range(999):
  x_train = np.append(x_train,[[xl[i],yl[i],tl[i]]],axis=0)

x_train.shape

# %%
# Create Model
model = keras.Sequential(
    [
        keras.Input(shape=(3)),
        keras.layers.Dense(100, activation="tanh"),
        keras.layers.Dense(3, activation= "sigmoid"),
    ]
)

model.summary()

# %%
# Compile 
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001), # choose optimizer and learning rate
    metrics=["accuracy"], # choose metrics
)

# %%
# Training
history = model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=2)

# %%
# Show loss and Accu
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('laccuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# %%
# Evaluate
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# %%
# Prediction
predictions = model.predict(x_test)


