const path = require('path');
const Max = require('max-api');
const fs = require('fs')
const tf = require('@tensorflow/tfjs-node');
const { assert } = require('console');
const constants = require('./constants.js');
const utils = require('./utils.js');
const { util } = require('@tensorflow/tfjs-node');
const { Midi } = require('@tonejs/midi'); // https://github.com/Tonejs/Midi

const UNIQUE_DRUM_VALUES = constants.UNIQUE_DRUM_VALUES;
const NUM_UNIQUE_DRUM_VALUES = UNIQUE_DRUM_VALUES.length;
const NUM_DRUM_CLASSES = constants.NUM_DRUM_CLASSES;
const NUM_STEPS = 64;

const MIN_PITCH_BASS = constants.MIN_PITCH_BASS;
const MAX_PITCH_BASS = constants.MAX_PITCH_BASS;
const NUM_BASS_PITCH = constants.NUM_BASS_PITCH;
const REST_PITCH_BASS = constants.REST_PITCH_BASS; // 61
const NOTEOFF_PITCH_BASS = constants.NOTEOFF_PITCH_BASS; // 62
const NUM_UNIQUE_BASS_VALUES = 64;//constants.NUM_UNIQUE_BASS_VALUES;
console.log("NUM_UNIQUE_BASS_VALUES", NUM_UNIQUE_BASS_VALUES);
console.log("REST_PITCH_BASS", REST_PITCH_BASS);
console.log("NOTEOFF_PITCH_BASS", NOTEOFF_PITCH_BASS);

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

function sample(probs, temperature) {
    return tf.tidy(() => {
      const logPreds = tf.div(tf.log(probs), Math.max(temperature, 1e-6));
      return tf.multinomial(logPreds, 1).dataSync()[0];
    });
}

async function generateBassline(drum_array, temperature){

    if (drum_array == null){
        utils.post("empty/null drum array?");
        drum_array =[ 6,  0,  0,  0,  0,  0,  0,  0, 34,  0,  0,  0,  0,  0,  0,  0,  6,  0,  0,  0,  0,  0,  0,  0,
                    34,  0,  0,  0,  0,  0,  0,  0,  6,  0,  0,  0,  0,  0,  0,  0, 34,  0,  0,  0,  0,  0,  0,  0,
                    6, 0,  0,  0,  0,  0,  0,  0, 34,  0,  0,  0,  0,  0,  0,  0];
    }
    let input_matrix = drumArrayToMatrix(drum_array);
    
    // Encode input drum pattern
    let input = tf.tensor2d(input_matrix, [NUM_STEPS, NUM_UNIQUE_DRUM_VALUES])
    input = tf.reshape(input_matrix, [1, NUM_STEPS, NUM_UNIQUE_DRUM_VALUES])
    let state = encoder.predict(input);

    // Empty input bassline
    let target_seq = tf.zeros([1, 1, NUM_UNIQUE_BASS_VALUES]);
    
    let output = [];
    for (let i = 0; i < NUM_STEPS; i++){
        let decoder_output = decoder.predict([target_seq, state[0], state[1]]);
        let yhat = decoder_output[0];
    
        let max_bass_value = tf.argMax(yhat, axis=2);
        let bass_value = sample(tf.squeeze(yhat), temperature);
        console.log(max_bass_value.dataSync(), bass_value);
        output.push(Number(bass_value));

        // # update state
        state = [decoder_output[1], decoder_output[2]];
        // # update target sequence
        target_seq = yhat
    }
    console.log(output);
    utils.post(output);
    
    // For sequencer output
    var pitch_sequence = [];
    var velocity_sequence = [];
    var duration_sequence = [];

    // output to max sequencer
    for (let i=0; i< NUM_STEPS; i++){
        let pitch = output[i];
        
        // how long this onset should long
        let duration_count = 1;
        let is_new_onset = true;
        if (isBassOnset(pitch)){
            for (let j=i+1; j <NUM_STEPS; j++){
                let new_pitch = output[j];
                // note off or note-on of other pitch
                if (!isBassOnset(new_pitch) || new_pitch == NOTEOFF_PITCH_BASS) break;
                if (new_pitch == pitch) duration_count++;
                else break;
            }
            if (i > 0){
                let prev_pitch = output[i - 1];
                if (prev_pitch == pitch) is_new_onset = false;                
            }
            if (is_new_onset){
                // normalize duration to 0 - 127
                let duration = Math.floor((duration_count * 8))
                pitch_sequence.push(pitch + MIN_PITCH_BASS)
                velocity_sequence.push(100);  // constant value
                duration_sequence.push(duration);
            } else {
                pitch_sequence.push(0)
                velocity_sequence.push(0);
                duration_sequence.push(0);
            }
        } else{
            pitch_sequence.push(0)
            velocity_sequence.push(0);
            duration_sequence.push(0);
        }    
    }
    assert(pitch_sequence.length == NUM_STEPS && velocity_sequence.length == NUM_STEPS && duration_sequence.length == NUM_STEPS)

    Max.outlet("pitch_output", 1, pitch_sequence.join(" "));
    Max.outlet("velocity_output", 1, velocity_sequence.join(" "));
    Max.outlet("duration_output", 1, duration_sequence.join(" "));
    Max.outlet("generated", 1); 
}

function isBassOnset(pitch){
    return (pitch < REST_PITCH_BASS);
}

// Generate a rhythm pattern
Max.addHandler("generate", ()=>{
    generateBassline();
});

// Start encoding... reset input matrix
var input_onset;
Max.addHandler("encode_start", (is_test) =>  {
    Max.post("encode_start");
    input_onset     = utils.create2DArray(NUM_STEPS, NUM_DRUM_CLASSES);
});

Max.addHandler("encode_add", (pitch, time, duration, velocity, muted, mapping) =>  {

    // select mapping
    let midi_map = constants.MAGENTA_MIDI_MAP;;

    // add note
    if (!muted){
        var unit = 0.25; // 1.0 = quarter note   grid size = 16th note 
        const half_unit = unit * 0.5;
        const index = Math.max(0, Math.floor((time + half_unit) / unit)) // centering 
        Max.post("index", index, pitch);
        if (index < NUM_STEPS){
            if (pitch in midi_map){
                let drum_id = midi_map[pitch];
                Max.post("pitch", pitch, drum_id);
                input_onset[index][drum_id]     = 1;
            } else {
                console.log("MIDI note pitch not found", pitch)
            }
        } 
    }
});

Max.addHandler("encode_done", () =>  {
    utils.post(input_onset);
    
    let drum_array = [];
    for (let i=0; i < NUM_STEPS; i++){
        let step_onset = []; // onsets in the step
        for (let j=0; j < NUM_DRUM_CLASSES; j++){
            if (input_onset[i][j] > 0) step_onset.push(j);
        }
        let drumComboId = constants.getDrumComboId(step_onset);
        drum_array.push(drumComboId);
        // console.log("step/drumcombo", step_onset, drumComboId);
    }
    
    // // Encoding!
    // var inputOn     = tf.tensor2d(input_onset, [NUM_STEPS, NUM_DRUM_CLASSES]);
    // output encoded z vector
    generateBassline(drum_array);
});


function isValidMIDIFile(midiFile){
    if (midiFile.header.tempos.length > 1){
        utils.error("not compatible with midi files containing multiple tempo changes")
        return false;
    }
    return true;
}

function getTempo(midiFile){
    if (midiFile.header.tempos.length == 0) return 120.0 // no tempo info, then use 120.0 
    return midiFile.header.tempos[0].bpm;  // use the first tempo info and ignore tempo changes in MIDI file
}

// Get location of a note in pianoroll
function getNoteIndexAndTimeshift(note, tempo){
    var unit = 0.25; // 1.0 = quarter note   grid size = 16th note 
    const half_unit = unit * 0.5;

    const index = Math.max(0, Math.floor((note.time + half_unit) / unit)) // centering 
    const timeshift = (note.time - unit * index)/half_unit; // normalized

    return [index, timeshift];
}


// Convert midi into pianoroll matrix
function processPianoroll(midiFile, midi_map){
    const tempo = getTempo(midiFile);

    let drum_input = utils.create2DArray(NUM_STEPS, NUM_DRUM_CLASSES);

    midiFile.tracks.forEach(track => {
    
        //notes are an array
        const notes = track.notes
        notes.forEach(note => {
            if ((note.midi in midi_map)){

                let timing = getNoteIndexAndTimeshift(note, tempo);
                let index = timing[0];
                
                if (index < NUM_STEPS){
                    let drum_id = midi_map[note.midi];
                    Max.post("pitch", note.midi, drum_id);
                    drum_input[index][drum_id]     = 1;
                } 
            }
        })
    })

    /*    for debug - output pianoroll */
    // if (velocities.length > 0){ 
    //     var index = utils.getRandomInt(velocities.length); 
    //     let x = velocities[index];
    //     for (var i=0; i< NUM_DRUM_CLASSES; i++){
    //         for (var j=0; j < LOOP_DURATION; j++){
    //             Max.outlet("matrix_output", j, i, Math.ceil(x[i][j]));
    //         }
    //     }
    // }

    return drum_input;
}

Max.addHandler("encode_midi", (filename, temperature = 0.1) => {
    utils.post("encode_midi", filename);

    // // Read MIDI file into a buffer
	try {
    	var input = fs.readFileSync(filename);
	} catch(e) {
		utils.post( "encode_midi error", e.message );
		return; 
	}

    var midiFile = new Midi(input);  
    if (isValidMIDIFile(midiFile) == false){
        utils.error("Invalid MIDI file: " + filename);
        return false;
    }

    // select mapping
    let midi_map = constants.MAGENTA_MIDI_MAP;

    // process midifile
    let drum_input = processPianoroll(midiFile, midi_map);
    utils.post(drum_input);
    
    let drum_array = [];
    for (let i=0; i < NUM_STEPS; i++){
        let step_onset = []; // onsets in the step
        for (let j=0; j < NUM_DRUM_CLASSES; j++){
            if (drum_input[i][j] > 0) step_onset.push(j);
        }
        let drumComboId = constants.getDrumComboId(step_onset);
        drum_array.push(drumComboId);
        // console.log("step/drumcombo", step_onset, drumComboId);
    }
    
    // // Encoding!
    // var inputOn     = tf.tensor2d(input_onset, [NUM_STEPS, NUM_DRUM_CLASSES]);
    // output encoded z vector
    generateBassline(drum_array, temperature);
    return true;
});