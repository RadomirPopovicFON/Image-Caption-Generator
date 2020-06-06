# Image-Caption-Generator
In this project I wanted to investigate, does using pre-trained features within a simple deep learning architecture can produce reliable results on image caption/tagging task. For this purpose, I used pre-trained embeddings [Fasttext](https://fasttext.cc/) and image features from [ResNet](https://keras.io/api/applications/resnet/). [Flickr8k](https://www.kaggle.com/shadabhussain/flickr8k?select=Flickr_Data) dataset was being used for training purposes.


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
    <td><i>a dog runs through the grass</i></td>
    <td><i>a man in red shorts is playing on the back</i></td>
  </tr>
  <tr>
    <td>10</td>
    <td><i>a basketball player in uniform is ready to throw the player in the air</i></td>
    <td><i>football player in red sooners ready to throw the player</i></td>
    <td><i>a small dog is jumping over a small dog</i></td>
    <td><i>a man with a mohawk and a yellow shirt is laying on the back</i></td>
  </tr>
  <tr>
    <td>40</td>
    <td><i>basketball players are tackling basketball and one is surfer</i></td>
    <td><i>football player sooners football</i></td>
    <td><i>a dog is running through a forest</i></td>
    <td><i>a man with a orange hat and a brown jacket has a cigarette on his beach</i></td>
  </tr>
 </table>
<br/>
<i>It may be ambious to conclude whether we had success with this particular model :). In general, quality of the output, along side model hyperparameters, depends on the actual images we include in training: Model will hardly output text of an entity which hasn't seen priorly in the data. </i><br/><br/><i>Furthermore, as the number of epochs increases, corresponding captions may not appear to be lineraly more accurate. In fact, quality may decrease sometimes. One of the reasons for such behavior is the inability of the model to continue to reduce the error rate. (Model will be updated in the future*)</i>
