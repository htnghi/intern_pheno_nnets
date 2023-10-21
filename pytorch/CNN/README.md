# Basic knowledge of CNN
A CNN works very similar to how our human eye works. A convolutional neural network (CNN) takes an input image and classifies it into any of the output classes. 
Image Classification is the technique to extract the features from the images to categorize them in the defined classes

* CNN Architecture: Each image passes through a series of different layers – primarily **convolutional layers, pooling layers, and fully connected layers**. 

* **Convolutional layers**: the core function behind a CNN. It is multiplication of the image matrix with a filter matrix to extract some important features from the image matrix.
    - It is a mathematical operation between the input image and the kernel (filter). The filter is passed through the image and the output is calculated as follows
    - Different filters are used to extract different kinds of features. 
    - => Adding more convolutional layers => increases the depth of the output layer => leads to increasing the number of parameters that the network needs to take care of. \
    => This increase in network dimensionality: increases in time and space complexity of the mathematical operations that take place in the learning process. 

* **Pooling layer**: used to reduce the size of any image while maintaining the most important features. 
    - The most common types: are max and average pooling => which take the max and the average value respectively from the given size of the filter => helps us in reducing the number of features i.e. it sharpens them so that our CNN performs better.
    -  The goal: is to subsample => shrink the input (while perserve important information) in order to reduce computational load, the memory usage, and the number of parameters => reduce the risk of overfitting
    - A common practice: add pooling layers after every one or two convolutional layers in the CNN architecture.

* **Activation function**:
    - ReLU: To all of the convolutional layers we apply the RELU activation function.
    - Linear: While mapping the convolutional layers to the output (called the fully connected layers)
    - Sigmoid: final activation
    
* Addition: **Dropout layer** is placed in between the fc layers and this randomly drops the connection with a set probability which will help us in training the CNN better.

# The first dataset (CIFAR-10)
* Dataset CIFAR-10: has 60,000 color images (RGB) at 32 x 32 pixels belonging to 10 different classes (6000 images/class). The dataset is divided into 50,000 training and 10,000 testing images.
    - The images are of size 32x32x3

* When training model, the loss is slightly decreasing with more and more epochs. This is a good sign. 
    - But loss fluctuate at the end => which could mean the model is overfitting or that the batch_size is small.
