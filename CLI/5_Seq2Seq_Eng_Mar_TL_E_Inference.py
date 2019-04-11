# Required packages
import pandas as pd
import numpy as np
import string
import re
import pickle
import os

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from string import digits


def clean_text(sentence:list=None, language:str=None):
    """
        Input: List, String
        Output: List
        Takes in text as list of strings. Returns text cleaned for NMT purposes.
    """
    if language == None:
        print("Please enter which language.")
        return None
        
    exclude = set(string.punctuation)
    remove_digits = str.maketrans('', '', string.digits)
        
    if language == 'en':
        sentence = sentence.lower()
        sentence = ''.join(ch for ch in sentence if ch not in exclude)
        sentence = sentence.translate(remove_digits)
        sentence = sentence.strip()
        sentence = re.sub(" +", " ", sentence)
        return sentence
    
    elif language == 'hi':
        sentence = sentence.lower()
        sentence = ''.join(ch for ch in sentence if ch not in exclude)

        sent_temp = ''
        for c in sentence:
            if c == ' ':
                sent_temp += c
            elif ord(u'\u0900') <= ord(c) <= ord(u'\u097F'):
                sent_temp += c
        sentence = sent_temp
      
        sentence = re.sub('[a-z]', '', sentence)
        sentence = re.sub('[०१२३४५६७८९।]', '', sentence)
        sentence = sentence.translate(remove_digits)
        sentence = sentence.strip()
        sentence = re.sub(" +", " ", sentence)
        return sentence
    
    elif language == 'ma':
        sentence = sentence.lower()
        sentence = ''.join(ch for ch in sentence if ch not in exclude)
        sentence = re.sub('[a-z]', '', sentence)
        sentence = re.sub('[०१२३४५६७८९।]', '', sentence)
        sentence = sentence.translate(remove_digits)
        sentence = sentence.strip()
        sentence = re.sub(" +", " ", sentence)
        return sentence
    
    else:
        print("Language not found")
        return None


def encode_input(X):
    """
        X = batch of inputs
    """
    # Get the batch_size
    batch_size = len(X)
    
    # Create a numpy array of zeros to hold input
    encoder_input_data = np.zeros((batch_size, max_length_src), dtype='float32')
    
    for i, input_text in enumerate(X):
        for t, word in enumerate(input_text.split()):
            if word not in source_dictionary.keys():
                word = 'UNK'
            encoder_input_data[i, t] = source_dictionary[word]
            
    return encoder_input_data


def encode_target(y):
    """
        y = batch of outputs
    """
    # Get the batch_size
    batch_size = len(y)
    
    # Create numpy arrays of zeros to hold encoded targets
    decoder_input_data = np.zeros((batch_size, max_length_tar), dtype='float32')
    decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens), dtype='float32')
    
    for i, target_text in enumerate(y):
        for t, word in enumerate(target_text.split()):
            if t < len(target_text.split()) - 1:
                decoder_input_data[i, t] = target_dictionary[word]
                
            if t > 0:
                decoder_target_data[i, t-1, target_dictionary[word]] = 1.0
                
    return decoder_input_data, decoder_target_data


def generate_batch(X, y, batch_size=128):
    """
        X = Source dataset
        y = Target dataset
        batch_size = Size of each batch
    """
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = encode_input(X[j:j+batch_size])
            decoder_input_data, decoder_target_data = encode_target(y[j:j+batch_size])
            
            yield([encoder_input_data, decoder_input_data], decoder_target_data)



def decode_sequence(input_sequence):
    # Get thought vector by encoding the input sequence
    states_value = inference_encoder.predict(input_sequence)
    
    # Generate target sequence initialized with <START> character
    target_sequence = np.zeros((1, 1))
    target_sequence[0, 0] = target_dictionary['<START>']
    
    # To stop the recurrent loop
    stop_condition = False
    
    # Final sentence
    decoded_sentence = ''
    
    while not stop_condition:
        # Get next prediction
        output_tokens, h, c = inference_decoder.predict([target_sequence] + states_value)
        
        # Get the token with max probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_reverse_dictionary[sampled_token_index]
        decoded_sentence += ' ' + sampled_word
        
        # Test for exit condition
        if (sampled_word == '<END>') or (len(decoded_sentence) > 50):
            stop_condition = True
            
        # Update the target sequence with current prediction
        target_sequence = np.zeros((1, 1))
        target_sequence[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    return decoded_sentence



print("Initializing model...")

# Mounted directory
root = '.'

# Define maximum lengths of source and target sentences.
max_length_src = 34
max_length_tar = 37

#### Embeddings ####
print("Loading embeddings...")


# Get vocabulary and embeddings
with open(os.path.join(root, 'embeddings.en'), 'rb') as f:
    english_summary = pickle.load(f)
    
with open(os.path.join(root, 'embeddings.ma'), 'rb') as f:
    marathi_summary = pickle.load(f)



# Add start and end tokens to dictionary
for word in ['<START>', '<END>']:
    l = len(marathi_summary['dictionary'].keys())
    marathi_summary['dictionary'][word] = l
    marathi_summary['reverse_dictionary'][l] = word
    marathi_summary['embeddings'] = np.vstack((marathi_summary['embeddings'], np.zeros((1, marathi_summary['embeddings'].shape[1]))))
####


#### Vocabulary ####
print("Creating vocabulary...")


# English vocabulary
all_eng_words = set(list(english_summary['dictionary'].keys()))

# Marathi vocabulary
all_mar_words = set(list(marathi_summary['dictionary'].keys()))


num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_mar_words)
num_encoder_tokens, num_decoder_tokens


source_dictionary = english_summary['dictionary']
target_dictionary = marathi_summary['dictionary']


source_reverse_dictionary = english_summary['reverse_dictionary']
target_reverse_dictionary = marathi_summary['reverse_dictionary']
####

#### Define model ####
print("Creating model...")

latent_dim = 128
batch_size = 128
epochs = 50

#### Encoder ####
# Inputs
encoder_inputs = Input(shape=(None, ), name='Encoder_Inputs')

# Embedding Lookup
encoder_embedding_layer = Embedding(num_encoder_tokens, latent_dim, mask_zero=True, name='English_Embedding_Layer')
encoder_embeddings = encoder_embedding_layer(encoder_inputs)

# LSTM
encoder_lstm = LSTM(latent_dim, return_state=True, name='Encoder_LSTM')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embeddings)

# Keeping only the states and discarding encoder outputs
encoder_states = [state_h, state_c]
####

#### Decoder ####
# Decoder Inputs
decoder_inputs = Input(shape=(None, ), name='Decoder_Inputs')

# Embedding
decoder_embedding_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero=True, name='Marathi_Embedding_Layer')
decoder_embeddings = decoder_embedding_layer(decoder_inputs)

# LSTM
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='Decoder_LSTM')
decoder_outputs, _, _ = decoder_lstm(decoder_embeddings, initial_state=encoder_states)

# Dense output layer
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='Decoder_Dense')
decoder_outputs = decoder_dense(decoder_outputs)
####


# Define a model with these layers
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Load pre-trained weights from Eng-Hin model
model.load_weights(os.path.join(root, 'best_model_en_ma_tl_e.hdf5'), by_name=True)


#### Inference model ####
# Encoder model to create a thought vector from the input
inference_encoder = Model(encoder_inputs, encoder_states)

# For each time step, the decoder states from previous timestep would act as inputs
decoder_state_input_h = Input(shape=(latent_dim, ), name='Inference_Decoder_Output')
decoder_state_input_c = Input(shape=(latent_dim, ), name='Inference_Decoder_Memory')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Embedding
decoder_embeddings_inference = decoder_embedding_layer(decoder_inputs)

# LSTM
decoder_outputs_inference, state_h_inference, state_c_inference = decoder_lstm(decoder_embeddings_inference, initial_state=decoder_states_inputs)
decoder_states_inference = [state_h_inference, state_c_inference]

# Dense
decoder_outputs_inference = decoder_dense(decoder_outputs_inference)

# Decoder model
inference_decoder = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inference] + decoder_states_inference
)
####



if __name__ == '__main__':
	# Condition to exit loop
	keep_looping = True

	while keep_looping:
		# Get input sentence
		sentence = input("Input (in English): ")

		# Clean sentence
		sentence = clean_text(sentence, 'en')

		# Encode input sentence
		input_sequence = encode_input([sentence])

		# Get output sentence
		decoded_sentence = decode_sequence(input_sequence)
		
		# Remove end of line token.
		output_sentence = ' '.join(decoded_sentence.split()[:-1])

		# Print output
		print("Output (in Marathi):", output_sentence)

		# Ask to continue or exit.
		check = input("Enter more? (y/n) ").lower()

		if check == 'n':
			keep_looping = False

	print("Thank you for using this translator!")

