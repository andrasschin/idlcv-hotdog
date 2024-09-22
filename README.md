## Create a python environment
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt

## Run the train.py script
python train.py --channels 32 64 128 128 --kernels 7 6 3 3 --strides 2 2 2 1 --img-size 128 
where the channels/kernels/strides describe the parameter for the given conv layer in the network



- Project hotdog_nothotdog

- The tasks

- Design and train a CNN to do the classification task, evaluate its performance, and document the process. 

    - • How did you decide on your architecture? Did you start out with something else? How/why did you decide to change it?

    - • How did you train it? Which optimizer did you use? Did you compare using different optimizers? Did you do other things to improve your training? 

    - • Did you use any data augmentation? Did you check if the data augmentation improved performance? 

    - • Did you use batch normalization? Does it improve the performance? 

    - • What is the accuracy of your network? Which test images are classified wrong? Any of them for obvious reasons? 

    - • Did you use transfer learning? Does it improve the performance? 

    - • Do you have a model that performs well? Try computing the saliency maps for some images of hotdogs. 
