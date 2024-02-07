Hi! This is Chui Ting Cheung.

The packages in the environment that I used is here in requirements.txt in the folder. (FYI, I used conda)

To train and test the model, simple run command "python main.py". 
The output should look something like in Result_Screenshot.png, except I changed a bit of the printing format the output.

Short description of files:
- main.py: prepares data and embeddings for training, dev and testing
- model.py: contains the class of the LSTM model
- train.py: trains, evaluates and tests the model with the data and embeddings from main.py
- data/: folder that contains train, dev and test datasets
- embeddings/: folder that contains pretrained glove word embeddings

Here are some sources I refered to while doing this task:
- PyTorch: Example: An LSTM for Part-of-Speech Tagging:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging

- fyyc: Kaggle: first try: lstm with glove by pytorch
https://www.kaggle.com/code/fyycssx/first-try-lstm-with-glove-by-pytorch/notebook

- NAZMUL TAKBIR: Kaggle: NLP with PyTorch - GloVe, LSTM, Ensembling
https://www.kaggle.com/code/nazmultakbir/nlp-with-pytorch-glove-lstm-ensembling#Downloading-and-Processing-GloVe-Files