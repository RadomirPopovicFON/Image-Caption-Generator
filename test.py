import argparse
import numpy as np
import pickle as pkl

from keras.preprocessing import image
from keras_preprocessing.sequence import pad_sequences
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import Model
from keras.models import load_model


def get_image_caption(img_path, tk, params):

    resnet50 = ResNet50(weights='imagenet')
    resnet_model = Model(inputs=resnet50.input, outputs=resnet50.layers[-2].output)
    text = 'startseq'
    tk.index_word[0] = '' #padding from keras tokenizer

    for i in range(1, params['MAX_LEN']):

        sequence = [tk.word_index[w] for w in text.split() if w in tk.word_index]
        sequence = np.reshape(np.array(pad_sequences([sequence[0:i]], padding='post', maxlen=params['MAX_LEN'])[0]),
                              (1, params['MAX_LEN']))

        img = image.load_img(img_path, target_size=(224, 224))
        x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
        encoded_img = resnet_model.predict(x)
        softmax_predictions = model.predict([encoded_img, sequence], verbose=0).flatten()
        prediction = np.argmax(softmax_predictions)

        word = tk.index_word[prediction]
        text += ' ' + word

        if word == 'endseq':
            break

    return text.replace('startseq ','').replace(' endseq', '')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Give hyperparameters for the model.')
    parser.add_argument('img_path', action="store", type=str, help='Path of image we want to generate caption.')
    parser.add_argument('model_path', action="store", type=str, help='Path of model we trained.')

    img_path = parser.parse_args().img_path
    model_path = parser.parse_args().model_path

    #_______________________________________________________________________________________________________________#

    caption_data = pkl.load(open("pkl_files/caption_data.pkl", "rb"))
    tk = caption_data['tk']
    params = caption_data['params']
    model = load_model(model_path)

    print(get_image_caption(img_path, tk, params))