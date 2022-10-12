# Help Protect the Great Barrier Reef

### Abstract
The project is about *object detection* of crown-of-thorns starfish in underwater image data. The reference Dataset has been taken from the Kaggle Challenge [https://www.kaggle.com/competitions/tensorflow-great-barrier-reef]. <br/> 

### Implementation
After properly preprocessing the dataset, we have exploited **RetinaNet** architecture with a ResNet50-FPN as backbone, pretrained on coco 2017. In order to obtain a coherent prediction, we performed a *fine-tuning* on our data of the last stage of the backbone architecture
and the remaining RetinaNet structure (Anchor generator, classification head and detection
head). We trained for 10 epochs with batch size equal to 4 and a learning rate
equal to 1e-5. During training we also added as a regularization technique a random horizontal flip with probability 0.5.

### Results
We have obtained good results, reaching an Average Precision of almost 0.9 in the validation set. We also tried to test our model with unseen images downloaded from the web to understand in a better way its performances.
<br/>
<p align="center">
  <img src="https://github.com/lolloloschi97/tensorflow-great-barrer-reef/blob/main/image3.jpeg" width=70% height=70% class="center"><br/>
</p>
<br/>
We can observe that not all crown-of-thorns starfish are correctly recognized and also in some cases, the bounding boxes were not completely covering all the starfish but just a portion of them. Here there are some examples:
<br/>
<br/>
<p align="center">
  <img src="https://github.com/lolloloschi97/tensorflow-great-barrer-reef/blob/main/image2.jpeg" width=30% height=30% class="center">
  <img src="https://github.com/lolloloschi97/tensorflow-great-barrer-reef/blob/main/image1.jpeg" width=30% height=30% class="center">
</p>

### Contribution
- Lorenzo Loschi
- Irene Rachele Lavopa
