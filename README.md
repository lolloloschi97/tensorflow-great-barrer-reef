# Help Protect the Great Barrier Reef

### Abstract
The project is about *object detection* of crown-of-thorns starfish in underwater image data. The reference Dataset has been taken from the Kaggle Challenge [https://www.kaggle.com/competitions/tensorflow-great-barrier-reef]. <br/> 

### Implementation
After properly preprocessing the dataset, we have exploited **RetinaNet** architecture with a ResNet50-FPN as backbone, pretrained on coco 2017. In order to obtain a coherent prediction, we performed a *fine-tuning* on our data of the last stage of the backbone architecture
and the remaining RetinaNet structure (Anchor generator, classification head and detection
head). We trained for 10 epochs with batch size equal to 4 and a learning rate
equal to 1e-5. During training we also added as a regularization technique a random horizontal flip with probability 0.5.

### Results
