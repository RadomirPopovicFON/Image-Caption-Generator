# Image-Caption-Generator
Generating captions for images using deep learning model with pre-trained embeddings [Fasttext](https://fasttext.cc/) and image features from [ResNet](https://keras.io/api/applications/resnet/). [Flickr8k](https://www.kaggle.com/shadabhussain/flickr8k?select=Flickr_Data) dataset was being used for training purposes.

Brief pipeline description:
1. Firstly, actual captions from images are saved in tokenized, model-ready format. Additionally, embedding matrix is generated from Fasttext pretrained weights. [See notebook here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/caption_extraction.ipynb) 
2. Secondly, images are converted into ResNet50 features. [See notebook here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/image_extraction.ipynb)
3. Model is being trained by combining embedding and resnet50 features. [See .py file here](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/train.py)
4. Model is being tested. [See .py file here](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/test.py)


