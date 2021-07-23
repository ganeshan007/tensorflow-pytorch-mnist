import sys
import os
import tensorflow as tf
mnist_data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_data.load_data()
x_train, x_test = x_train/255.0, x_test/255.0


net = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(400,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])
print(net.summary())
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   #Since we are using softmax so we don't have logits but a probability distribution
net.compile(optimizer='adam',metrics=['accuracy'],loss=criterion)
net.fit(x_train,y_train,epochs=10)

