const path = require('path');
const Max = require('max-api');
const fs = require('fs')
const tf = require('@tensorflow/tfjs-node');
const { assert } = require('console');

const UNIQUE_DRUM_VALUES = require('./constants.js').UNIQUE_DRUM_VALUES;
const NUM_UNIQUE_DRUM_VALUES = UNIQUE_DRUM_VALUES.length;
const NUM_UNIQUE_BASS_VALUES = 64;
const NUM_STEPS = 64;

let decoder, encoder;

Max.post(`Loaded the ${path.basename(__filename)} script`);

async function loadModels(){
    Max.post('Start loading models...');
    decoder = await tf.loadLayersModel("file://" + path.resolve("../models_tfjs/decoder/model.json"));
    encoder = await tf.loadLayersModel("file://" + path.resolve("../models_tfjs/encoder/model.json"));
}
loadModels();

function create2DArray(row, col){
    var x = new Array(row);
    for (var i = 0; i < x.length; i++) {
        x[i] = new Array(col);
        for (var j =0; j < x[i].length; j++){
            x[i][j] = 0.0;
        }
    }
    return x;
}

function drumArrayToMatrix(input){
    let matrix = create2DArray(NUM_STEPS, NUM_UNIQUE_DRUM_VALUES);
    assert(input.length == NUM_STEPS)
    for (let i=0; i < input.length; i++){
        let drum_value = UNIQUE_DRUM_VALUES.indexOf(input[i]);
        matrix[i][drum_value] = 1;
    }
    return matrix;
}

async function generateBassline(){

    let test_input =[ 6,  0,  0,  0,  0,  0,  0,  0, 34,  0,  0,  0,  0,  0,  0,  0,  6,  0,  0,  0,  0,  0,  0,  0,
    34,  0,  0,  0,  0,  0,  0,  0,  6,  0,  0,  0,  0,  0,  0,  0, 34,  0,  0,  0,  0,  0,  0,  0,
     6, 0,  0,  0,  0,  0,  0,  0, 34,  0,  0,  0,  0,  0,  0,  0];
    let test_matrix = drumArrayToMatrix(test_input);
    
    // Encode input drum pattern
    let input = tf.tensor2d(test_matrix, [NUM_STEPS, NUM_UNIQUE_DRUM_VALUES])
    input = tf.reshape(input, [1, NUM_STEPS, NUM_UNIQUE_DRUM_VALUES])
    let state = encoder.predict(input);

    // Empty input bassline
    let target_seq = tf.zeros([1, 1, NUM_UNIQUE_BASS_VALUES]);
    
    let output = [];
    for (let i = 0; i < NUM_STEPS; i++){
        let decoder_output = decoder.predict([target_seq, state[0], state[1]]);
        let yhat = decoder_output[0];
        
        let bass_value = tf.argMax(yhat, axis=2);
        output.push(Number(bass_value.dataSync()));

        // # update state
        state = [decoder_output[1], decoder_output[2]];
        // # update target sequence
        target_seq = yhat
    }
    console.log(output);

}

// Generate a rhythm pattern
Max.addHandler("generate", ()=>{
    generateBassline();
});