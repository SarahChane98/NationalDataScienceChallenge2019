import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import pickle
# create a dictionary of {words:integer value} so that each unique word is represented by a number
# this dict includes all words from train train set

dense_layers = [0]
layer_sizes = [16]
drop_out = [0.2]



def to_onehot_n(inds, dim):
    ret = np.zeros((len(inds), dim))
    ret[np.arange(len(inds)), inds] = 1
    return ret


dim = 58

train_data = np.load('processed_data/train_data.npy')
train_label = to_onehot_n(np.load('processed_data/train_label.npy').astype(np.int), dim=dim)
#print(type(text_dict))  # need to convert numpy ndarray back to dictionary !!!!
pickle_in = open('processed_data/text_dict.pickle', "rb")
text_dict = pickle.load(pickle_in)




# what is the max length of title
train_length = []
for i in range(len(train_data)):
    train_length.append(len(train_data[i]))

print(max(train_length))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=text_dict["<PAD>"],
                                                        padding='post',
                                                        maxlen=36)

vocab_size = len(text_dict)

for drop_out in drop_out:
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            NAME = "{}-dropout-{}-nodes-{}-dense-{}-noBN".format(drop_out, layer_size, dense_layer, int(time.time()))
            tensorboard = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME))

            model = keras.Sequential()
            model.add(keras.layers.Embedding(vocab_size, layer_size))
            #model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.GlobalAveragePooling1D())

            for l in range(dense_layer):
                model.add(keras.layers.Dense(layer_size, activation=tf.nn.relu))
                model.add(keras.layers.Dropout(drop_out))

            model.add(keras.layers.Dense(58, activation=tf.nn.softmax))
            model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['acc'])

            model.fit(train_data, train_label, epochs=100, batch_size=512, validation_split=0.3, callbacks=[tensorboard])
            model.save('trained_models/{}.model'.format(NAME))