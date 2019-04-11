# Example-Based-Neural-Machine-Translation-System
A Word-level Sequence-to-Sequence Neural Machine Translation system. It uses an Encoder-Decoder architecture, where both the encoder and decoder are Recurrent Neural Networks that use LSTM cells. 

## Objective
This project has been created by keeping in mind those Indian languages and dialects for which bilingual parallel corpus, when available, is of negligible size. In such cases, transfer learning has been found to show improvement.
Refer to:
  1) https://aclweb.org/anthology/D16-1163.pdf
  2) https://aclweb.org/anthology/W18-6325.pdf
  
## Approach
An Encoder-Decoder NMT system has been first trained on English-Hindi parallel dataset. The learnt layer weights have then been transfered to another Encoder-Decoder NMT model, which has been trained using English-Marathi parallel dataset. The English-Hindi dataset has been comparatively larger as compared to the English-Marathi dataset.

## Improvements
Transfer learning has been one step forward to improve the accuracy of prediction model. Furthermore, Skip-gram Word2Vec embeddings have been used to further improve the accuracy of the system.

## Datasets
  1) English-Hindi parallel corpus (68,000 lines): Anoop Kunchukuttan, Pratik Mehta, Pushpak Bhattacharyya. The IIT Bombay English-Hindi Parallel Corpus. Language Resources and Evaluation Conference. 2018.
  2) English-Hindi parallel corpus (2000 lines): http://www.manythings.org/anki/
  3) English-Marathi parallel corpus (33000 lines): http://www.manythings.org/anki/
  
## Files
  1) 1_Data_Cleaning.ipynb
    Script to preprocess textual dataset by cleaning irrelevant symbols, punctuations and digits from English, Hindi and Marathi dataset files.
    
  2) 2_Gather_Dataset.ipynb
    Script to join all available text in each language (Hindi, English and Marathi) into a single file for input to a Word2Vec model for training embeddings.
    
  3) 3_Word2Vec.ipynb
    A model built from scratch in Tensorflow to train a Skip-gram Neural Word Embeddings.
    
  4) 4_*.ipynb
    Code files to train each model (Eng-Hin and Eng-Mar) with / without transfer learning and with / without Neural Word Embeddings.
    NTL -> No transfer learning
    TL  -> Transfer learning
    E   -> Embeddings
    NE  -> No embeddings
    
  5) 5_Seq2Seq_Eng_Mar_TL_E.ipynb
    Code for an inference model that loads pre-trained weights for actual use case.
    
## Command Line Interface
  A command line interface has been created for anyone who wishes to try this system out of the box. To get it working, follow these steps:
    1) Download CLI folder. This folder contains the interface, along with the trained model weights and Word2Vec embeddings.
    2) Install dependencies by running:
          pip install -r requirements.txt
    3) Run the 5_Seq2Seq_Eng_Mar_TL_E_Inference.py file using Python3.6:
          python 5_Seq2Seq_Eng_Mar_TL_E_Inference.py
          
## References
  1) Word2Vec: Deep Learning by Google at Udacity.com
  2) Encoder-Decoder architecture: https://towardsdatascience.com/word-level-english-to-marathi-neural-machine-translation-using-seq2seq-encoder-decoder-lstm-model-1a913f2dc4a7
  
## If using this code for public / private, commercial / non-commercial purposes, kindly give a shout-out to my repository.
