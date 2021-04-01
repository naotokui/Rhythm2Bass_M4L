#%%

import seq2seq_LSTM
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.utils import to_categorical
from IPython.display import SVG

%pylab inline

#%%

BASS_PADDING = 63

#%%
import keras
import tensorflow as tf 

print(keras.__version__)
print(tf.__version__)


#%%
import numpy as np

data = np.load("../../data/tv_themes__magenta__min_pitch_24__max_pitch_84__q_note_res_4__infer_chords_True__n_bars_4__max_cont_rests_16__max_N_inf.npz", allow_pickle=True)
triplets = data['triplets']

# %%

bassline = []
drums = []

for t in triplets:
    # if len(t['b'].keys()) > 1:
    #     print("multiple bass") 
    for k in t['b'].keys():
        b = t['b'][k]
        bassline.append(b[0])
    drums.append(t['d'])

bassline = np.array(bassline)
drums = np.array(drums)

print(bassline.shape, drums.shape)
# %%

np.savez("../data/tv_thems_bassline_drums_only.npz", bassline=bassline, drums=drums)
# %%

bassline = np.load("../data/tv_thems_bassline_drums_only.npz")['bassline']
drums = np.load("../data/tv_thems_bassline_drums_only.npz")['drums']
print(bassline.shape, drums.shape)
# %%

unique_bass = np.unique(bassline)
unique_bass = np.append(unique_bass, BASS_PADDING)
num_output = len(unique_bass) # output dimention
print(unique_bass, num_output)

unique_drums = np.unique(drums)
num_input = len(unique_drums) # input dimention
print(unique_drums, num_input)


# %%

# mapped - remove unused values from sequences

def convert_sequence(seqs, unique_values):
    output = []
    for seq in seqs:
        out = [list(unique_values).index(s) for s in seq]
        output.append(out)
    output =  np.array(output)
    return to_categorical(output, num_classes=len(unique_values))

def get_input_bassline(seqs):
    output = []
    for seq in seqs:
        flipped = np.flip(seq, 0)
        flipped = np.append(BASS_PADDING, flipped[:-1])
        output.append(flipped)
    return np.array(output)


bassline_input = get_input_bassline(bassline)
bassline_input_mapped = convert_sequence(bassline_input, unique_bass)
bassline_mapped = convert_sequence(bassline, unique_bass)
drums_mapped = convert_sequence(drums, unique_drums)
print(bassline_mapped.shape, drums_mapped.shape)


# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#%%
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
# %%
def define_models(n_input, n_output, n_units):
    '''
     a generic function to define an encoder-decoder
    recurrent neural network
    :param n_input: The cardinality of the input sequence, e.g. number of features, words, or characters for each time step.
    :param n_output: The cardinality of the output sequence, e.g. number of features, words, or characters for each time step.
    :param n_units: The number of cells to create in the encoder and decoder models, e.g. 128 or 256.
    :return:
        train: Model that can be trained given source, target, and shifted target sequences.
        inference_encoder: Encoder model used when making a prediction for a new source sequence.
        inference_decoder Decoder model use when making a prediction for a new source sequence.
    '''
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

N_UNITS = 128 

model, encoder, decoder = define_models(n_input=num_input, n_output=num_output, n_units=N_UNITS)
# %%
model.summary()
encoder.summary()
decoder.summary()
# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# %%
history = model.fit([drums_mapped, bassline_input_mapped], bassline_mapped, epochs=1000)
# %%

model.save("../tmp/model.h5")
encoder.save("../tmp/encoder.h5")
decoder.save("../tmp/decoder.h5")

# %%
import random

drum_input = random.choice(drums_mapped)
drum_input = np.expand_dims(drum_input, 0)

print(drum_input.shape)

# encode
state = encoder.predict(drum_input)

# start of sequence input
target_seq = np.array([0.0 for _ in range(num_output)]).reshape(1, 1, num_output)
print(target_seq.shape)

# collect predictions
output = list()


# %%
n_steps = 64
for t in range(n_steps):
    # predict next char
    yhat, h, c = decoder.predict([target_seq] + state)
    # store prediction
    output.append(yhat[0, 0, :])
    # update state
    state = [h, c]
    # update target sequence
    target_seq = yhat
output = np.array(output)
# %%

print(np.argmax(output, axis=1).shape)

print(output.shape)
# %%

for step in range(output.shape[0]):
    b = output[step]
    print(b)
    index = np.argmax(b)
    bass_pitch = unique_bass[index]
    print(bass_pitch)
# %%

np.argmax(random.choice(bassline_mapped), axis=1)
# %%

# %%
