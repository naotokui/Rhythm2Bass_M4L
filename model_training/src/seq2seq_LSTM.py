from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
from random import *
from keras.utils import plot_model, to_categorical
from copy import deepcopy
import os
from music21 import *


# source https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
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


def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    '''
    The function can be used after the model is trained to
    generate a target sequence given a source sequence.
    :param infenc: Encoder model used when making a prediction for a new source sequence.
    :param infdec: Decoder model use when making a prediction for a new source sequence.
    :param source: Encoded source sequence.
    :param n_steps: Number of time steps in the target sequence.
    :param cardinality: The cardinality of the output sequence, e.g.
            the number of features, words, or characters for each time step.
    :return:
        output: predicted sequence
    '''

    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0, 0, :])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    return np.array(output)


def get_sequences(bassline_dataset_file, drum_dataset_file):
    # get bassline dataset
    bassline_dataset = np.loadtxt(bassline_dataset_file, dtype="int")

    # get drum dataset
    with open(drum_dataset_file, "r") as f:
        drum_text = f.read()
        drum_dataset_0b = []
        drum_dataset_int = []

        for drum_pattern in drum_text.split("\n"):
            if drum_pattern:
                drum_dataset_0b.append(drum_pattern.split("\t"))
                temp = []
                for pattern in drum_pattern.split("\t"):
                    temp.append(int(pattern, 2))
                drum_dataset_int.append(temp)

    return drum_dataset_0b, drum_dataset_int, bassline_dataset


def get_drum_sequences(drum_dataset_file, loop_length_bars=2):
    # get drum dataset
    with open(drum_dataset_file, "r") as f:
        drum_text = f.read()
        drum_dataset_0b = []
        drum_dataset_int = []

        for drum_pattern in drum_text.split("\n"):
            if drum_pattern:
                drum_dataset_0b.append(drum_pattern.split("\t"))
                temp = []
                for pattern in drum_pattern.split("\t"):
                    temp.append(int(pattern, 2))
                print("temp", len(temp))
                drum_dataset_int.append(temp)

    return drum_dataset_0b, drum_dataset_int


def create_input_output_sequences(bassline_dataset_file, drum_dataset_file, begin_char="-"):
    drum_dataset_0b, drum_dataset_int, bassline_dataset = get_sequences(bassline_dataset_file, drum_dataset_file)

    encoder_input = drum_dataset_int
    decoder_input = []
    decoder_output = []

    for ix, bassline in enumerate(bassline_dataset):
        bassline = bassline.tolist()
        temp = []
        for timestep in range(len(bassline)):
            if timestep == 0:
                temp.append(begin_char)
            else:
                temp.append(bassline[timestep - 1])
        decoder_input.append(temp)
        decoder_output.append(bassline)

    return np.array(encoder_input), np.array(decoder_input), np.array(decoder_output)


def get_dataset(drum_dataset_int, bassline_dataset, padding_value="S", input_cardinality=256, output_cardinality=256):
    X1, X2, y = list(), list(), list()
    # cardinality_src = len(np.unique(drum_dataset_int))+1
    # cardinality_tar = len(np.unique(bassline_dataset))+1
    # cardinality_tar2 = len(np.unique(bassline_dataset))+1
    source_words_tokens = get_words_tokens(drum_dataset_int)
    target_words_tokens = get_words_tokens(bassline_dataset.tolist())
    target_in_words_tokens = get_words_tokens(bassline_dataset.tolist(), padding_value)
    for ix in range(len(drum_dataset_int)):
        # generate source sequence
        source_Vectorized = Vectorize(drum_dataset_int[ix], source_words_tokens)
        # define target sequence
        target = bassline_dataset[ix].tolist()
        target_Vectorized = Vectorize(target, target_words_tokens)
        target.reverse()
        # create padded input target sequence
        target_in = [padding_value] + target[:-1]
        target_in_Vectorized = Vectorize(target_in, target_in_words_tokens)
        # encode
        src_encoded = to_categorical([source_Vectorized], num_classes=input_cardinality)
        tar_encoded = to_categorical([target_Vectorized], num_classes=output_cardinality)
        tar2_encoded = to_categorical([target_in_Vectorized], num_classes=output_cardinality)

        # store
        X1.append(src_encoded[0])
        X2.append(tar2_encoded[0])
        y.append(tar_encoded[0])
    return np.array(X1), np.array(X2), np.array(y), source_words_tokens, target_words_tokens, target_in_words_tokens


def get_words_tokens(dataset, padding=None):
    flat_dataset = []
    for entry in dataset:
        flat_dataset.extend(entry)
    if not padding is None:
        flat_dataset.extend(padding)
    words = set(flat_dataset)
    tokens = list(range(1, len(words) + 1))
    words_tokens = dict()
    for (word, token) in zip(words, tokens):
        words_tokens[word] = token

    return words_tokens


def Vectorize(x_array, words_tokens):
    x_tokenized = []
    for x in x_array:
        type(x)
        x_tokenized.append(words_tokens[x])

    return x_tokenized


def one_hot_decode(encoded_seq):
    return [np.argmax(Vector) for Vector in encoded_seq]


def sample_vector(probability_vector, sample_mode=0, temperature=1):
    '''

    :param probability_vector: Output probabilities of the one hot encoded outputs
    :param sample_mode: Mode 0: Select the highest probable output
                        Mode 1: Select based on the probability distribution (as is, not scaled)
                        Mode 2: Scale the distribution using temperature and output most probable one
    :param temperature:
                        Used for scaling the distribution
    :return:
    '''
    sampled_index = None
    if sample_mode == 0:  # takes the highest possible value
        sampled_index = np.argmax(probability_vector)
    if sample_mode == 1:
        indices = range(len(probability_vector))
        sampled_index = np.random.choice(indices, 1, p=probability_vector)[0]
    if sample_mode == 2:
        # helper function to sample an index from a probability array
        a = np.asarray(probability_vector).astype('float64')
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        sampled_index = np.argmax(np.random.multinomial(1, a, 1))
    if sample_mode == 3: # filters output probability
        print("before", sum(probability_vector), probability_vector )
        probability_vector = 1/probability_vector
        probability_vector = probability_vector/(sum(probability_vector)+.1)
        print("after", sum(probability_vector), probability_vector)
        a = np.asarray(probability_vector).astype('float64')
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        sampled_index = np.argmax(np.random.multinomial(1, a, 1))
        #sampled_index = np.argmax(probability_vector)

    return sampled_index


def one_hot_to_bassline(encoded_seq, words_tokens, sample_mode=0, temperature=1):
    # returns the actual bassline from the one-hot encoded outputs
    tokens_words = {v: k for k, v in words_tokens.items()}
    #print("tokens_words", tokens_words)
    bassline = []

    for Vector in encoded_seq:
        sampled_ix_from_vector = sample_vector(Vector, sample_mode=sample_mode, temperature=temperature)
        #print("sampled_ix_from_vector", sampled_ix_from_vector)
        #print("KEYSKEYS", list(tokens_words.keys()))
        if sampled_ix_from_vector in list(tokens_words.keys()):
            bassline.append(tokens_words[sampled_ix_from_vector])
        else:
            bassline.append(tokens_words[1])
    return bassline


def bassline_to_midi(bassline_array, filename, note_duration=.25):
    noteStream = stream.Stream()

    start_times = []
    midi_values = []

    # create filename directory if doesn't exist
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    for time_stamp, midi_number in enumerate(bassline_array):
        if midi_number != 0:
            start_times.append(time_stamp)
            midi_values.append(midi_number)

    print(start_times, midi_values)
    grid_lines = list(range(len(bassline_array)))

    for ix, grid_line in enumerate(grid_lines):

        if grid_line in start_times:
            note_index = start_times.index(grid_line)
            midi_value = midi_values[note_index]
            print("midi_value", midi_value)
            note_class = midi2note(midi_value)
            myNote = note.Note(note_class)

            print("midi_value", midi_value, "myNote.pitch", myNote.pitch)
            myNote.duration.quarterLength = note_duration

        else:
            myNote = note.Rest()
            myNote.duration.quarterLength = note_duration

        noteStream.append(myNote)
        noteStream.write("midi", filename)
    return noteStream


def midi2note(midi_number):
    # converts midi number to note name
    pitch_classes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

    octave = str(int(np.floor(midi_number / 12) - 1))
    pitch_class = pitch_classes[int(midi_number % 12 - 9)]
    note = pitch_class + octave

    return note


if __name__ == '__main__':
    # bassline_to_midi([0,24,0,0], filename="temp.mid", note_duration=.25)
    drum_testset_0b, drum_testset_int = get_drum_sequences(drum_dataset_file="../data_test/drum_size_5.txt")
    k = 2
    drum_dataset_int_words_tokens = {
        0: 1, 128: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 15: 10, 16: 11, 143: 12,
        23: 13, 24: 14, 152: 15, 31: 16, 47: 17, 48: 18, 56: 19, 63: 20, 200: 21, 207: 22, 79: 23,
        216: 24, 120: 25, 95: 26, 224: 27, 223: 28, 127: 29,
        240: 30, 248: 31, 250: 32, 251: 33, 252: 34, 253: 35, 255: 36}

    print(drum_dataset_int_words_tokens)

    source_Vectorized = Vectorize(drum_testset_int[k], drum_dataset_int_words_tokens)
