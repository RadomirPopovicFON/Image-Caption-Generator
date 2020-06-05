import argparse
import pickle as pkl
import numpy as np

from keras.layers import GRU, Embedding, Dense, Dropout, BatchNormalization
from keras.layers.merge import add
from keras.models import Model
from keras import Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def data_generator(encoded_imgs, image_captions, batch_threshold):

    train_X_img, train_X_text, train_Y, i = [], [], [], 0

    while True:
        for (image_name, image_value) in encoded_imgs.items():
            i += 1
            for sent in image_captions[image_name]:

                for len_sent, val in enumerate(sent): # length of padded sentence (found through first 0 index)
                    if val == 0: break

                for j in range(1, len_sent):
                    train_X_img.append(image_value)
                    train_X_text.append(pad_sequences([sent[0:j]], padding='post', maxlen=params['MAX_LEN'])[0])
                    train_Y.append(to_categorical([sent[j]], num_classes=params['NB_WORDS'])[0])

            if i == batch_threshold:
                yield [[np.array(train_X_img), np.array(train_X_text)], np.array(train_Y)]
                i = 0
                train_X_img, train_X_text, train_Y = [], [], []

def model_generator():

    input_img = Input(shape=(2048,), name='input_img')
    drop_img = Dropout(0.5, name='drop_img')(input_img)
    dense_img = Dense(512, activation='elu', name='dense_img')(drop_img)
    dense_norm_img = BatchNormalization(name='dense_norm_img')(dense_img)

    input_txt = Input(shape=(params['MAX_LEN'],), name='input_txt')
    emb_txt = Embedding(params['NB_WORDS'], params['EMB_DIM'], mask_zero=True, name='emb_txt')(input_txt)
    drop_txt = Dropout(0.5, name='drop_txt')(emb_txt)
    lstm_txt = GRU(512, activation='elu', name='lstm_txt')(drop_txt)
    lstm_norm_txt = BatchNormalization(name='lstm_norm_txt')(lstm_txt)

    addition = add([dense_norm_img, lstm_norm_txt], name='addition')
    dense_full = Dense(512, activation='elu', name='dense_full')(addition)
    outputs = Dense(params['NB_WORDS'], activation='softmax', name='outputs')(dense_full)

    model = Model(inputs=[input_img, input_txt], outputs=outputs)
    model.get_layer('emb_txt').set_weights([emb_matrix])
    model.get_layer('emb_txt').trainable = False
    model.summary()

    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Give hyperparameters for model.')
    parser.add_argument('epochs', action="store", type=int, help='Number of epochs.')
    parser.add_argument('no_imgs_per_batch', action="store", type=int, help='Number of images per batch.')

    epochs = parser.parse_args().epochs
    no_imgs_per_batch = parser.parse_args().no_imgs_per_batch

    #_________________________________________________________________________________________________________________#

    encoded_imgs = pkl.load(open("pkl_files/encoded_imgs.pkl", "rb"))
    caption_data = pkl.load(open("pkl_files/caption_data.pkl", "rb"))
    image_captions = caption_data['image_captions']
    emb_matrix = caption_data['emb_matrix']
    params = caption_data['params']

    dataset_size = len(image_captions)
    steps = dataset_size // no_imgs_per_batch
    model = model_generator()
    model.compile(loss='categorical_crossentropy', optimizer='Adadelta')

    for i in range(epochs):
        generator = data_generator(batch_threshold=no_imgs_per_batch,
                                   encoded_imgs=encoded_imgs, image_captions=image_captions)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save('weights/model_' + str(i) + '.h5')