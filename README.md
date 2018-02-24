# Tensorflow Tutorial

This is a basic tutorial on how to use TensorFlow to build a convolutional neural network for classifying handwritten digits. For this project, we are using the MNIST dataset with 55,000 training images and 10,000 test images.

This project has two branches:
1. `master`: This branch contains template code for our TF application. Some lines of code will need to be added to get this working.
2. `solutions`: This branch contains the full runnable code.

## Getting Started
Before you start, clone or download this repository onto your machine. We will be running our program in `Python 2.7` so we need to make sure that `Python 2.7` is installed on your machine along with `pip`, the package manager we will be using for installing the rest of the packages we will be using in this tutorial.

**Windows**: You will need to download `Python 2.7` from [this link](https://www.python.org/downloads/windows/). This installation should automatically install `pip` for you, so you don't have to worry about that.

**Mac**: You should already have `Python 2.7` installed (by default) on your machine. To install `pip`, type the following commands into `Terminal`:
```sh
$ curl https://bootstrap.pypa.io/get-pip.py > get-pip.py
$ sudo python get-pip.py
```

**Mac and Windows**: At this point, you should have both `Python 2.7` and `pip` installed on your machine. Next, we will be installing `virtualenv`, a virtual environment package that will help us install libraries specifically for this tutorial. `virtualenv` will ensure that our new libraries will not interfere with our system and we will be able to easily enter/exit our environment to enable/disable the packages that we install within it.

To install `virtualenv`, type the following command into `Terminal` or `cmd`:
```sh
$ pip install virtualenv
```

Then, navigate to the directory of the project you are working on (e.g., tensorflow-tutorial/) and run the following command:
```sh
$ virtualenv env
```

This will create a new folder called `env` which will hold all of the data about our new Python virtual environment. This folder will hold information about any packages that we install while the `env` is active; these packages will only be active or usable while the `env` is active. This is our desired result.

**Mac**: To activate the environment, type the following command within the same directory as the `env` folder you just created:
```sh
$ source env/bin/activate
(env) $
```
Now, you can see the `(env)` text show up before the command prompt, indicating that you have the environment active. To deactivate the environment, simply enter `(env) $ deactivate`.

**Windows**: To activate the environment, type the following command within the same directory as the `env` folder you just created:
```sh
$ .\env\Scripts\activate
(env) $
```
Now, you can see the `(env)` text show up before the command prompt, indicating that you have the environment active. To deactivate the environment, simply enter `(env) $ deactivate`.

**Mac and Windows**: Now that we have created our virtual environment, the rest of the installation is straightforward. We need to make sure that we have the packages `tensorflow`, `opencv-python`, and `scipy` installed for the code to run properly. To do so, enter the following commands in the `env`:
```sh
(env) $ pip install tensorflow
(env) $ pip install opencv-python
(env) $ pip install scipy
```

Now, you should have all of the necessary dependencies installed! You can do a quick test of this by trying the following imports inside the `Python` interpreter:
```sh
(env) $ python
>>> import tensorflow
>>> import cv2
>>> import scipy
>>> import numpy
```

If you get no errors here, your setup should be ready to go (note: press Ctrl-D to end the interpreter)!

## Writing the code
In the `master` branch, some lines of code have been omitted for you to fill out. These lines are indicated with the `### YOUR CODE ###` tags. These missing lines are the most essential components of building and running the tensorflow code. The code itself is commented throughout to help explain different lines of code, but I will explain a few notable ones here.
```python
def cnn_model_fn(features, labels, mode):
    ...
```

The above function is where the actual neural network is created. Here we will build our network to have two convolutional layers, each followed by a max-pooling layer. After the convolutional layers, we will flatten our data and run it through two fully-connected layers to perform the final classification. Our model inputs will be the raw pixels of an image of a digit (0-9) and our output will be a vector of probabilities designating the probability that the input was a 0, a 1, a 2, etc. Finally, we take the highest probability to be our prediction.

If we are currently training our model, we will also want to define the loss and function we want to use to optimize loss. Then, `tensorflow` will feed our training data through the model and try to minimize loss (a way to determine error or goodness-of-fit). As long as we link together the model correctly and define the inputs, loss optimization, and outputs, `tensorflow` will run our model and train it for us. This function will return a `tf.estimator.EstimatorSpec` object which contains the information to run the model.

In the `main()` method, we really leverage the utility of `tensorflow` to do a lot of the low-level work for us. First, we download the dataset. Then, we create the model according to the `cnn_model_fn()` function. Then, we specify how the inputs should be fed into the model and tell `tensorflow` to train our model. Finally, we can specify how the test data will be fed into the model and tell `tensorflow` to evaluate the test accuracy of our model.

The function `evaluate_my_handwritten()` can be used to have the model actually predict your own handwritten digits. To use this function, you just need to take a picture of your handwritten digit, crop it so that the only marks seen in the image are part of your actual digit (no stray marks), then place the raw images into a folder named `new-data/data/raw`. Then, set the value of `MY_DATA_FOLDER = 'new-data/data'` so that the program knows where to find your digits. Finally, make sure that each image filename begins with the digit that is shown in the image (e.g., an image of the digit `8` could be called `8_digit.jpg`). An example of handwritten digits is provided in `new-data.tar` which will be automatically extracted for you if you do not create your own directory called `new-data/`.

## Useful Links
* [Andrew Ng's Coursera](https://www.coursera.org/learn/machine-learning)
* [Intuitive explanation of convolutional neural networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
* [Chapter on ConvNets (part of an online TensorFlow tutorial)](https://www.safaribooksonline.com/library/view/learning-tensorflow/9781491978504/ch04.html)
* Datasets to play with at [this link](https://deeplearning4j.org/opendata):
    * Street View House Numbers (SVHN)
    * ImageNet
    * Google Open Images
* [Kaggle has a lot of machine learning datasets and competitions](https://www.kaggle.com)
* [ModelZoo is a curated list of many different machine learning models](https://github.com/BVLC/caffe/wiki/Model-Zoo)
