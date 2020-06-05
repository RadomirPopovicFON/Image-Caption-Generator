# Image-Caption-Generator
Generating captions for images using deep learning model with pre-trained embeddings [Fasttext](https://fasttext.cc/) and image features from [ResNet](https://keras.io/api/applications/resnet/). [Flickr8k](https://www.kaggle.com/shadabhussain/flickr8k?select=Flickr_Data) dataset was being used for training purposes.


### Brief pipeline description:
* Firstly, actual captions from images are saved in tokenized, model-ready format. Additionally, embedding matrix is generated from Fasttext pretrained weights. [See notebook here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/caption_extraction.ipynb) 
* Secondly, images are converted into ResNet50 features. [See notebook here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/image_extraction.ipynb)
* Model is being trained by combining embedding and resnet50 features. After each epoch, weights are being saved and are available for testing. As arguments one can pass number of epochs, number of images per batch. Additionally, one can change other model hyperparameters within the code directly. [See .py file here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/train.py)
* Model can be tested for a given image and model path. [See .py file here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/test.py)<br/>


### Model used along with test output:
<p align="center">
  <img src="https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/images/model.png" width="500">
</p>

Results after 1st epoch:

<table>
  <tr>
    <td><img src="https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/images/test-1.jpg" width=200 height=150></td>
    <td><img src="https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/images/test-2.jpg" width=200 height=150></td>
    <td><img src="https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/images/test-3.jpg" width=150 height=200></td>
    <td><img src="https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/images/test-4.jpg" width=150 height=200></td>
  </tr>
  <tr>
    <td><i>a basketball player in the air to catch the ball</i></td>
    <td><i>a person in a grassy area </i></td>
    <td><i>a dog with a little girl in a grassy yard</i></td>
    <td><i>a person in a blue shirt is standing in the air</i></td>
  </tr>
 </table>
