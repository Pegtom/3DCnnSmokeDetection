# 3DCnnSmokeDetection

I used a parallel 3dCNN model based on https://github.com/dipakkr/3d-cnn-action-recognition for action recognition using the UCF101 dataset. I had to modify this code due to my custom dataset. You can use the UCF101 dataset as well via the modified code.
Because of the high amount of data, I faced many errors, such as memory allocation and lack of ram space. As a result, I had to modify the code and use generators to produce batches of frames in the pipeline. 

# Options
Options of 3dcnn.py are as follows:

--batch batch size

--epoch the number of epochs

--videos a name of the directory where the dataset is stored

--nclass the number of classes you want to use

--output a directory where the results described above will be saved

--color use RGB image or grayscale image

--skip getting frames at intervals or contenuously

--depth the number of frames to use


# Results
The validation accuracy of the fire dataset was 84.21 
