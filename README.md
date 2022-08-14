# Singer Recognition using Deep Learning

In this project, I implemented a dense neural network, a convolutional neural network and a recurrent neural network for singer recognition. The 4 singers with whom I trained the networks were Bruce Dickinson (Iron Maiden), Freddie Mercury (Queen), James Hetfield (Metallica) and Michael Jackson. This project was my final Bachelor's degree thesis, which can be found and read [here](https://zaguan.unizar.es/record/96551/files/TAZ-TFG-2020-2198.pdf) (spanish).

All audio tracks used in this project are 30 seconds long due to Copyright. By having the tracks this duration and using them for academic purposes, the well-known "Fair Use" is applied. "Fair use" is a doctrine in United States law that permits limited use of copyrighted material without having to first acquire permission from the copyright holder. For more information about this, you can refer to the official Copyright [website](https://www.copyright.gov/fair-use/more-info.html).



## Implementation

The file ``cfg.py`` contains the features to preprocess the audio before training the model, as well as the folders in which those models and the pickles will be saved once the model is trained.

The file ``eda.py`` performs the preprocessing of the audio tracks. In the ``singers.csv`` file are all names of all audio tracks that the model will use to train and their corresponding labels. Basically, the tracks are loaded from the ``wavfiles/`` folder, their sample rate is reduced from 48,000Hz to 16,000Hz in order to do the training faster, and the new audio tracks are stored in the ``clean/`` folder. Besides, an audio track of each singer is taken and their signal, their Fourier Transform, their Filter Bank Coefficients and their MFCC are plotted. For audio, the most common feature to work with in deep learning is the MFCC (Mel frequency Cepstral Coefficient).

![signal](https://user-images.githubusercontent.com/71872419/147421132-96eeb031-de2a-4e21-802b-6bf70e57780a.png)

![fft](https://user-images.githubusercontent.com/71872419/147421143-57b64e59-b4ab-4b6f-b215-91afa438932c.png)

![fbc](https://user-images.githubusercontent.com/71872419/147421146-664d55d9-74ef-462d-9323-0118969601c0.png)

![mfcc](https://user-images.githubusercontent.com/71872419/147421149-ff57af51-3dcc-47c8-afcc-ed35aa4255dc.png)

The ``model.py`` file contains the implementation of the three neural networks, dense, convolutional or recurrent. To select the one you want, you have to set the mode in the line 144 to "feedforward", "conv" or "time", depending on the neural network you want to implement, respectively. The models and pickles will be saved in the ``models/`` and ``pickles/`` folders, and once the training is finished, the loss and accuracy curves will be plotted. For example, for the CNN network, I got these curves:

<img src="https://user-images.githubusercontent.com/71872419/147421193-09dfe7db-50d9-4250-a831-2c27e5d97d3d.png" width="408" height="300">   <img src="https://user-images.githubusercontent.com/71872419/147421196-54641f2f-458c-43ec-8834-9729bd10be70.png"  width="408" height="300">

Finally, the file ``predict.py`` contains the implementation for making predictions. In the ``singers_test.csv`` file are all names of all audio tracks that will be used to test the model. First, the audio tracks to be predicted, contained in the ``wavfiles_ToPredict/`` folder, are preprocessed in the same way as in the ``eda.py`` file. Once this preprocessing is done, the new tracks are stored in the ``clean_test/`` folder. Then, the model will be tested with these tracks, and the predictions will be saved in a file like the ``Predictions_Example.csv`` one. It contains the name of the audio file, the original label, the probability that each singer is the one who sings on that track, and the singer with the largest probability.

![predictions](https://user-images.githubusercontent.com/71872419/147421229-0a13bf35-c236-464b-9535-1e0db797d266.PNG)



## Run the implementation

The first thing to do is to remove the ``README.md`` files from the ``clean/`` and ``clean_test/`` folders, they must be empty before running anything. Once this is done, open a terminal, go to the folder in which you have the project and run
```
python eda.py
```
The audio tracks will be processed and they will be ready to train the model. Now, in the file ``model.py``, set the mode mentioned above in order to choose the type of neural network you want to train and then run
```
python model.py
```
The model and pickle will be saved. Finally, to make predictions just run 
```
python predict.py
```
and both the preprocessing of the audio tracks to be tested and the predictions will be carried out automatically.



## Possible modifications

If you want to try this networks with your own dataset, you should replace the wav files from the ``wavfiles/`` and ``wavfiles_ToPredict/`` folders with whatever you want, and also change the content of ``singers.csv`` and ``singers_test.csv``, adapting them to your dataset. Also, it is very likely that the number of labels you have in your dataset is different, so you will have to change the number of neurons in the output layer of the models.



## Citation
If you find this work useful in your research, please cite:
```
@thesis{FandosA,
      author = "Fandos, Andrés",
      title = "Reconocimiento de vocalistas mediante redes neuronales profundas",
      address = {Spain},
      school = "University of Zaragoza",                 
      type = "Bachelor's Thesis",            
      year = "2020"
}
```
or
> Fandos, A. “Reconocimiento de vocalistas mediante redes neuronales profundas”. Bachelor’s Thesis. University of Zaragoza, 2020.
