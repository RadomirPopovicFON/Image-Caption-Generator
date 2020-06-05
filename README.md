# Image-Caption-Generator
Generating captions for images using deep learning model with pre-trained embeddings [Fasttext](https://fasttext.cc/) and image features from [ResNet](https://keras.io/api/applications/resnet/). [Flickr8k](https://www.kaggle.com/shadabhussain/flickr8k?select=Flickr_Data) dataset was being used for training purposes.


###Brief pipeline description:
* Firstly, actual captions from images are saved in tokenized, model-ready format. Additionally, embedding matrix is generated from Fasttext pretrained weights. [See notebook here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/caption_extraction.ipynb) 

* Secondly, images are converted into ResNet50 features. [See notebook here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/image_extraction.ipynb)

* Model is being trained by combining embedding and resnet50 features. After each epoch weights are being saved and are available for testing. As arguments one can pass number of epochs, number of images per batch. Additionally, one can change other model hyperparameters within the code directly. ([See .py file here.])(https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/train.py)

* Model can be tested for a given image and model path. [See .py file here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/test.py)


####Model example and some of the test results:
<center><img src="https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/images/model.png" width="500"></center>


