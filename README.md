## Create a python environment
```python -m venv venv```

```venv\Scripts\activate.bat```

```pip install -r requirements.txt```

## Run the train.py script
```python train.py --channels 32 64 128 128 --kernels 5 3 3 3 --strides 1 1 1 1 --padding 2 1 1 1  --img-size 128```

where the channels/kernels/strides describe the parameter for the given conv layer in the network

## Data Loader 
`HotdogDataset class` handles loading the images and applying the transformations when the dataset is accessed.

```get_dataset``` function is designed to prepare and return a dataset with various transformations applied to the images. It takes several parameters that control the nature of these transformations, such as whether the dataset is for `training` (train), the desired `image size` (image_size), and several boolean flags that enable or disable specific transformations.Finally, an instance of the HotdogDataset class is created with the specified parameters and the composed transformation pipeline. 

## Project findings and conclusions

The tasks

- Design and train a CNN to do the classification task, evaluate its performance, and document the process. 

    - • How did you decide on your architecture? Did you start out with something else? How/why did you decide to change it?
    We started with a basic architecture in order to get things running and inspect our simple model performance. Here is the training-validation scores for accuracy!
    Epoch: [20]	Train Loss: 0.5075	Train Accuracy: 0.8003	Validation Loss: 0.5846	Validation Accuracy: 0.6958
    Channels: [32, 64, 128, 128]
    Kernels: [7, 6, 3, 3]
    Strides: [2, 2, 2, 1]
    Image Size: 128
    ----------------------
    [0] Layer resolution: 61x61
    [1] Layer resolution: 28x28
    [2] Layer resolution: 13x13
    [3] Layer resolution: 11x11

    With this simple architecture we scored 
 

    - • How did you train it? Which optimizer did you use? Did you compare using different optimizers? Did you do other things to improve your training? 
        We firstly train it without any data augmentation and preprocessing. We only resized the images and begin the process. We used Adam Optimizer.

    - • Did you use any data augmentation? Did you check if the data augmentation improved performance? 
        Yes....

    - • Did you use batch normalization? Does it improve the performance? 
        Yes...

    - • What is the accuracy of your network? Which test images are classified wrong? Any of them for obvious reasons? 
        This....Yes there are classified wrong because...

    - • Did you use transfer learning? Does it improve the performance? 

    - • Do you have a model that performs well? Try computing the saliency maps for some images of hotdogs. 
