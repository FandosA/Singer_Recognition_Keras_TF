# Singer Recognition using Deep Learning

In this project, I implemented a dense neural network, a convolutional neural network and a recurrent neural network for singer recognition. The 4 singers with whom I trained the networks were Bruce Dickinson (Iron Maiden), Freddie Mercury (Queen), James Hetfield (Metallica) and Michael Jackson. This project was my final bachelor's degree thesis, which can be found and read [here](https://zaguan.unizar.es/record/96551/files/TAZ-TFG-2020-2198.pdf) (spanish).


## Implementation

The file ``cfg.py`` contains the features to preprocess the audio before training the model, as well as the folders in which those models and the pickles will be saved once the model is trained.

The file ``eda.py`` performs the preprocessing of the audio tracks. Basically, it loads the tracks from the ``wavfiles`` folder, reduces their sample rate from 48,000Hz to 16,000Hz in order to do the training faster, and store the new tracks in the ``clean`` folder. Besides, it takes an audio track of each singer and plots their signal, their Fourier Transform, their Filter Bank Coefficients and their MFCC.

The ``model.py`` file contains the implementation of the three neural networks, dense, convolutional or recurrent. To select the one you want, you have to set the mode in the line 144 to "feedforward", "conv" or "time", depending on the neural network you want to implement, respectively. The models and pickles will be saved in the ``models`` and ``pickles`` folders, and once the training is finished, the loss and accuracy curves will be plotted.

Finally, the file ``predict.py`` contains the implementation for making predictions. First, the audio to be predicted is preprocessed, in the same way as in the ``eda.py`` file. The tracks I used to make predictions are in the ``wavfiles_ToPredict`` folder, and once this preprocessing is done, the new tracks are stored in the ``clean_test`` folder. Then, the model will be tested with these tracks, and the predictions will be saved in a file like the ``Predictions_Example.csv`` one. It contains the name of the audio file, the original label, the probability that each singer is the one who sings on that track, and the singer with the larger probability.

![predictions](https://user-images.githubusercontent.com/71872419/147421038-cdde5354-f0cc-4e1d-b0b7-13fb8a4a4e76.PNG)



## Citation
If you find this work useful in your research, please cite:
```
@thesis{FandosA,
  author = "Fandos, Andrés and Civera, Javier",
  title = "{Reconocimiento de vocalistas mediante redes neuronales profundas}",
  school = {University of Zaragoza},
  type = {Bachelor's Thesis},
  year = "2020",
}
```
or
> Fandos, A. and Civera, J. “Reconocimiento de vocalistas me-diante redes neuronales profundas”. Bachelor’s Thesis. University of Zaragoza, 2020.
