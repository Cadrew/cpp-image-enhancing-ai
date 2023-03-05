# C++ Image Enhancing with Tensorflow using ESRGAN

## Overview

This mockup is based on [CppFlow](https://github.com/serizba/cppflow) that uses [Tensorflow C API](https://www.tensorflow.org/install/lang_c) to run pre-trained models from [Real-ESRGAN project](https://github.com/xinntao/Real-ESRGAN).

## Prerequisites

Follow installation instructions here:
- [Tensorflow C API](https://www.tensorflow.org/install/lang_c)
- [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)


No need to install [CppFlow](https://github.com/serizba/cppflow) as it's already integrated inside the project.

## How it works

### Build

Run:
```bash
g++ -std=c++17 -o main.out main.cpp -ltensorflow -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs  
```
or
```bash
make build
```

### Models

Pre-trained models come from [Real-ESRGAN project](https://github.com/xinntao/Real-ESRGAN).
Models are located in the `models/` directory. The project needs `.pb` file (Tensorflow models) in order to work.

Get inputs and outputs info from the model, you need to install Tensorflow first:
```bash
saved_model_cli show --dir models/regular --tag_set serve --signature_def serving_default
```