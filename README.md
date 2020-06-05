# Image-Caption-Generator
Generating captions for images using a simple deep learning model with pre-trained embeddings [Fasttext](https://fasttext.cc/) and image features from [ResNet](https://keras.io/api/applications/resnet/). [Flickr8k](https://www.kaggle.com/shadabhussain/flickr8k?select=Flickr_Data) dataset was being used for training purposes.


### Brief pipeline description:
* Firstly, actual captions from images are saved in tokenized, model-ready format. Additionally, embedding matrix is generated from Fasttext pretrained weights. [See notebook here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/caption_extraction.ipynb) 
* Secondly, images are converted into ResNet50 features. [See notebook here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/image_extraction.ipynb)
* Model is being trained by combining embedding and resnet50 features. After each epoch, weights are being saved and are available for testing. As arguments one can pass number of epochs, number of images per batch. Additionally, one can change other model hyperparameters within the code directly. [See .py file here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/train.py)
* Model can be tested for a given image and model path. [See .py file here.](https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/test.py)<br/>


### Model used along with test output:
<p align="center">
  <img src="https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/images/model.png" width="500">
</p>

Results after 1st epoch (5 images per batch):

<table>
  <tr>
    <td>Epoch #</td>
    <td><img src="https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/images/test-1.jpg" width=250 height=200></td>
    <td><img src="https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/images/test-2.jpg" width=150 height=250></td>
    <td><img src="https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/images/test-3.jpg" width=150 height=250></td>
    <td><img src="https://github.com/RadomirPopovicFON/Image-Caption-Generator/blob/master/images/test-4.jpg" width=150 height=250></td>
  </tr>
  <tr>
    <td>1</td>
    <td><i>a basketball player in the air to catch the ball</i></td>
    <td><i>a football player in red uniform uniform red uniform uniform red uniform</i></td>
    <td><i>a dog with a little girl in a grassy yard</i></td>
    <td><i>a man in red shorts is playing on the back</i></td>
  </tr>
  <tr>
    <td>5</td>
    <td><i>a basketball player in blue is ready to hit the ball</i></td>
    <td><i>football player in red sooners</i></td>
    <td><i>a dog runs through a field</i></td>
    <td><i>a woman with a colorful mohawk is laying on the back</i></td>
  </tr>
 </table>

<i>As we can see, output may not be exactly correct, especially after 1st epoch, model can produce some funny results :). In general, quality of the utput, along side model hyperparameters, depends on the actual images we include in training. In other words, model will hardly output text of an entity which hasn't seen priorly in the data. (Will be further updated*)</i>
