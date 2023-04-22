<div align="center">
      <h1> Rotten Or Fresh Fruit Detection </h1>
      </div>


# Description
Detecting the rotten fruits become significant in the agricultural industry. Usually, the classification of fresh and rotten fruits is carried by humans is not effectual for the fruit farmers. Human beings will become tired after doing the same task multiple times, but machines do not. Thus, the project proposes an approach to reduce human efforts, reduce the cost and time for production by identifying the defects in the fruits in the agricultural industry. If we do not detect those defects, those defected fruits may contaminate good fruits. Hence, we proposed a model to avoid the spread of rottenness. The proposed model classifies the fresh fruits and rotten fruits from the input fruit images. In this work, we have used three types of fruits, such as apple, banana, and oranges

# Features
The recent approaches in computer vision, especially in the fields of machine learning and deep learning have improved the efficiency of image classification tasks [1-6]. Detection of defected fruits and the classification of fresh and rotten fruits represent one of the major challenges in the agricultural fields. Rotten fruits may cause damage to the other fresh fruits if not classified properly and can also affect productivity. Traditionally this classification is done by men, which was labor-intensive, time taking, and not efficient procedure. Additionally, it increases the cost of production also. Hence, we need an automated system which can reduce the efforts of humans, increase production, to reduce the cost of production and time of production.The recent approaches in computer vision, especially in the fields of machine learning and deep learning have improved the efficiency of image classification tasks [1-6]. Detection of defected fruits and the classification of fresh and rotten fruits represent one of the major challenges in the agricultural fields. Rotten fruits may cause damage to the other fresh fruits if not classified properly and can also affect productivity. Traditionally this classification is done by men, which was labor-intensive, time taking, and not efficient procedure. Additionally, it increases the cost of production also. Hence, we need an automated system which can reduce the efforts of humans, increase production, to reduce the cost of production and time of production.

# Tech Used
 ![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white) ![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white) ![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![Postman](https://img.shields.io/badge/Postman-FF6C37?style=for-the-badge&logo=postman&logoColor=white) ![Google Cloud](https://img.shields.io/badge/Google%20Cloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)
      
## Related work:
*A machine vision* system was developed for the detection of fruit skin defects in the study. Colour is the major feature used for categorization and a machine learning algorithm called Support Vector Machine (SVM) has been used in classification.

*Image processing* can help in the classification of the defect and non-defect fruits. It helps in identifying the defects on the surface of mango fruits. First, the fruits are collected manually and the researchers themselves classified them as fine and defected. Then pre-processing is carried out on the images and is given to a CNN model for the task of classification.

In our work, the proposed CNN model provides high accuracy in the classification task of fresh and rotten fruits. Here the proposed model’s accuracy is compared against the transfer learning models. Three types of fruits are selected from various types of fruits. The dataset is obtained from Kaggle where there are 13599 files with 6 classes.
Fresh apples
Fresh bananas
Fresh Orange
Rotten apples
Rotten bananas
Rotten Orange
We inspected the different pre-trained models of VGG16, VGG19, MobileNet, and Xception of transfer learning (transfer learning models). This paper introduces a powerful CNN model which has enhanced accuracy for fresh and rotten fruits classification task than transfer learning models while investigating the effect of very important hyperparameters to obtain better results and also avoid over-fitting.
## Methodology Proposed:
### 1. Dataset acquisition and pre-processing
The dataset is obtained from Kaggle which has three types of fruits-apples, bananas, and oranges with 6 classes i.e. each fruit divided as fresh and rotten. The total size of the dataset used in this work is 5989 images. The training images are of 3596, the validation set contains 596 images belongs to 6 classes, and the test set contains of 1797 images which belong to 6 classes.
 
### 2.Convolutional neural networks
In computers, the images are represented as related pixels. In the image, a certain collection of pixels may represent an edge in one image, some may represent the shadow of an image or some other pattern. One way to detect these patterns is by using convolution. During computation, the image pixels are represented using a matrix. For detecting the patterns, we need to use a “filter” matrix which is multiplied with the image pixel matrix. 

### 3.Proposed model for classification of fresh and rotten fruits
In our work, we proposed an accurate CNN model for classifying the fresh and rotten fruits which is shown in Fig 3. It consists of three convolutional layers. The first convolution layer uses 16 convolution filters with a filter size of 3x3, kernel regularizer, and bias regularizer of 0.05. It also uses random_uniform, which is a kernel initializer. It is used to initialize the neural network with some weights and then update them to better values for every iteration. Random_uniform is an initializer that generates tensors with a uniform distribution. Its minimum value is -0.05 and the maximum value of 0.05. Regularizer is used to add penalties on the layer while optimizing. These penalties are used in the loss function in which the network optimizes. No padding is used so the input and output tensors are of the same shape. The input image size is 224x224x3. Then before giving output tensor to max-pooling layer batch normalization is applied at each convolution layer which ensures that the mean activation is nearer to zero and the activation standard deviation nearer to 1.

### 4.Fresh and rotten fruits classification using transfer learning
Transfer learning takes what a model learns while solving one problem and applies it to a new application. Often it is referred to as ‘knowledge transfer’ or ‘fine-tuning’. Transfer learning consists of pre-trained models. Transfer learning releases few of the upper layers of a fixed model base and affixes new classifier layers and the final layers of the base model. This fine tuning of high level feature representations in the base model makes it applicable for the specific task.

## Highlights
- Batch Size - 32
- Accuracy - 96.90
- Channel- 3
- EPOCHS- 50
- Image Size-256*256



      
<!-- </> with 💛 by readMD (https://readmd.itsvg.in) -->
