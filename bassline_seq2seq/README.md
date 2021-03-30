# bassline_seq2seq

A generative model based on word-based sequence-to-sequence learning method. The purpose of this model is to generate a bassline given a drum loop.

# Dependancies:
1. Keras (with TensorFlow Backend)    https://keras.io/
2. Music21:                           http://web.mit.edu/music21/

Used for my thesis project, access report at: [WILL BE PUBLISHED LATER]

# Note:
The following tutorial was used in developing this project. Reviewing the tutorial is highly advised before using the scripts available in this project


Tutorial: https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/

# How To Use:
## Step 1: Prepare Data
The drum representation we propose is a list of time-quantized onset vectors in eight distinct frequency bands (B1 to B8): 40-70Hz, 70-110Hz, 130-145Hz, 160-190Hz, 300-400Hz, 5-7kHz, 7-10kHz and 10-15kHz.  Quantized to 16th-notetime steps and having a duration of 2 bars, the drums need to be represented at 32 different time steps. Hence, the drum representations used are a 32x8 matrix of onsets. On the other hand, the bassline is a vector of 32x1 where thenth entry identifies whether a note starts (or should be sustained) at the nth timestep. 

For every entry in the datasets, each bassline vector and each drum vector should be prepared based on the drum and bassline transcriptions described above, so as to be used for designing and implementing the models. In short, each bassline and drum vector should be prepared as follows:

1. Bassline Vector: Construct a bassline of length 32 (2 bars) where the ith element (0-31) in this vector represents the state of the bassline at the ith time step. Preferably, modify all the basslines so that they all start at the same pitch value. As an example, a typical bassline vector in the dataset looks like: array([48., 49.,  1000., ..., 48.,  1000.,  1000.,  0.]). A value of 1000 in this vector specifies that the note at the previous step needs to be sustained. A value of 0 in this vector identifies that there is a silence at the corresponding time step. Any other value in this vector is a MIDI number which identifies that a note (with the given MIDI number) exists at the corresponding time step.
    
2. Drums Vector: Construct a drums vector of length of 32 (2 bars) where the ith element in this vector represents the state of the drums at the ith time step. Each ith element of the drums vector is an 8-bit binary encoding in which the kth bit shows whether a drum onset exists in the kth frequecncy band at the ith time step . A typical drum vector for a sample loop in the dataset should look like: ['0b11111000', '0b00000000', '..., '0b00000011']. Note that the 0b prefix denotes the binary encoding. 

#### In this repository, a sample dataset is provided for testing the code. The dataset can be found at:
/data/bassline_size_50[OnsetsOnly]_translated_to_midi_36-2bars.txt
/data/bassline_size_50[WithOffet]_translated_to_midi_36-2bars.txt
/data/drum_size_50.txt

## Step 2: Train
Follow the steps provided in the following Jupyter notebook: /src/Train.ipynb

## Step 3: Predict
In the same manner as Step 1, create a text file containing the drum patterns for which a bassline should be predicted

Follow the steps provided in the following Jupyter notebook: /src/Predict.ipynb

#### In this repository, a sample test dataset is provided for testing the code. The dataset can be found at:
/data_test/drum_size_15.txt


