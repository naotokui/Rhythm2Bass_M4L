import os, glob
import numpy as np
import pickle

def get_bassline_drum_text_files(directory, bass_directory, drum_directory, loop_length_bars=2):
    print(directory)
    print(os.path.join(os.path.dirname(directory), "*"))
    releases = glob.glob(os.path.join(directory, "*"))
    bassline_files = []
    drum_files = []
    for release in releases:
        bassline_files.append(os.path.join(release, bass_directory))
        drum_files.append(os.path.join(release, drum_directory))
    return releases, bassline_files, drum_files



def translate_bassline_starting_note(bassline, firstMIDIValue = 48, hold_value=1000):
    # returns bassline starting from c2 (48) or a provided firstMIDIValue

    # find first note
    first_note = bassline[bassline>0][0]

    # calculate semitones
    for time_step, note in enumerate(bassline):
        if note>0 and note!=hold_value: # do not translate the hold_value
            bassline[time_step] = (bassline[time_step]-first_note)+firstMIDIValue

    return bassline

def bassline_txt_to_array(bassline_file, hold_value = 1000, translate_to=None,
                          loop_length_bars=2, duplicate_bassline=True):

    # hold_char denotes the sustained notes
    # exp. translate_to=48 the first note starts at 48 (C2)
    # loop_length_bars denotes the number of bars required for each loop
    # duplicate_bassline=True --> If the bassline is bar it will be repeated one more time otherwise will be
    #                               concatenated with zeros

    with open(bassline_file, "r") as f:
        print ("OPENING FILE: ", bassline_file)
        bassline_txt = f.read()

    print(bassline_txt)

    bassline_array_with_offset = np.zeros(loop_length_bars*16)
    bassline_array_without_offset = np.zeros(loop_length_bars*16)

    # read each midi event
    events = bassline_txt.split("\n")[1:]
    events = (event for event in events if event)   #remove empty lines

    for ix, event in enumerate(events): # event: onset\toffset\tmidi\tnote
        onset = int(event.split("\t")[0])
        offset = int(event.split("\t")[1])
        # print(event.split("\t"))
        if onset<=(loop_length_bars*16-2):
            bassline_array_with_offset[onset] = int(float(event.split("\t")[2]))
            bassline_array_without_offset[onset] = int(float(event.split("\t")[2]))

            if (offset-onset)>1:
                for i in range(1, offset-onset):
                    bassline_array_with_offset[onset+i]= hold_value

    annotated_bars_length = int(np.ceil(offset/16))

    for i in range(annotated_bars_length*16, loop_length_bars*16):
        bassline_array_with_offset[i] = bassline_array_with_offset[i-annotated_bars_length*16]
        bassline_array_without_offset[i] = bassline_array_without_offset[i-annotated_bars_length*16]

    #print("offset: ", offset, "annotated_bars_length", annotated_bars_length)
    print("bassline_file: ", bassline_file, len(bassline_array_with_offset))
    # print ("Last Note at: ", onset)

    if not translate_to is None:
        bassline_array_without_offset = translate_bassline_starting_note(bassline_array_without_offset, translate_to)
        bassline_array_with_offset = translate_bassline_starting_note(bassline_array_with_offset, translate_to, hold_value=hold_value)


    return bassline_array_with_offset, bassline_array_without_offset

def compile_basslines(releases, bassline_files,
                      compilation_name="../data/bassline",
                      translate_to=None,
                      loop_length_bars=2
                      ):
    # returns a list of arrays, each corresponding to a bassline
    # also saves a pickle with releases, bassline_files to keep track of the order of data
    basslines_without_hold_dataset = []
    basslines_with_hold_dataset = []

    for bassline_file in bassline_files:
        bassline_array_with_offset, bassline_array_without_offset = bassline_txt_to_array(bassline_file,
                                                                                          translate_to=translate_to,
                                                                                          loop_length_bars=loop_length_bars)

        # make sure the bassline has 32 events (32 time steps)
        '''
        if len(bassline_array_without_offset) < 32:
            bassline_array_without_offset = bassline_array_without_offset[:16]
            bassline_array_without_offset = np.repeat(bassline_array_without_offset, 1)
            bassline_array_with_offset = bassline_array_with_offset[:16]
            bassline_array_with_offset = np.repeat(bassline_array_without_offset, 1)
        '''
        basslines_without_hold_dataset.append(bassline_array_without_offset)
        basslines_with_hold_dataset.append(bassline_array_with_offset)
        # TBC following line produces error if translate_to is not None
        # basslines_with_hold_dataset.append(bassline_array_with_offset)


    if not translate_to is None:
        fname = compilation_name+"_size_"+str(len(basslines_without_hold_dataset))+\
                "_translated_to_midi_"+str(translate_to)+".txt"
        fname_without_offset = compilation_name + "_size_" + str(len(basslines_without_hold_dataset)) + \
                               "[OnsetsOnly]_translated_to_midi_" + str(translate_to) + ".txt"
        fname_with_offset = compilation_name + "_size_" + str(len(basslines_without_hold_dataset)) + \
                            "[WithOffet]_translated_to_midi_" + str(translate_to) + ".txt"
    else:
        fname = compilation_name + "_size_"+str(len(basslines_without_hold_dataset))+\
                "_translated_to_midi_" + str(translate_to) + ".txt"


        fname_without_offset = compilation_name + "_size_" + str(len(basslines_without_hold_dataset)) + \
                               "[OnsetsOnly]_translated_to_midi_" + str(translate_to) + ".txt"
        fname_with_offset = compilation_name + "_size_" + str(len(basslines_without_hold_dataset)) + \
                            "[WithOffet]_translated_to_midi_" + str(translate_to) + ".txt"

    np.savetxt(fname_without_offset, basslines_without_hold_dataset,
               fmt='%i', delimiter='\t', newline='\n')
    np.savetxt(fname_with_offset, basslines_with_hold_dataset,
               fmt='%i', delimiter='\t', newline='\n')

    with open(fname[:-4]+"_info.txt", "w") as f:
        f.write("release\tbassline\n")
        for ix, release in enumerate(releases):
            f.write("%s\t" % release)
            f.write("%s\n" % bassline_files[ix])

def compile_drums(releases, drum_files, compilation_name="../data/drum", loop_length_bars=2):
    '''

    :param releases: text files for the release tags of all drum annotations
    :param drum_files: text files for drum annotations
    :param compilation_name: name and location of the compiled dataset
    :return: None
    '''

    # create a list of "0b001.." annotations for each drum pattern and store in drum_dataset
    drum_dataset = []
    for drum_file in drum_files:
        drum_dataset.append(drum_txt_to_array(drum_file, loop_length_bars=loop_length_bars))

    # create a text file with the annotation of the drums on a separate line for each entry (drum pattern)
    text_to_save = ""
    for drum_pattern in drum_dataset:
        text_to_save+="\t".join(drum_pattern)
        text_to_save+="\n"

    fname = compilation_name + "_size_"+str(len(drum_files))+".txt"

    # Save results and the information regarding the results
    with open(fname, "w") as f:
        f.write(text_to_save)

    with open(fname[:-4]+"_info.txt", "w") as f:
        f.write("release\tdrum\n")
        for ix, release in enumerate(releases):
            f.write("%s\t" % release)
            f.write("%s\n" % drum_files[ix])


def drum_txt_to_array(drum_file, loop_length_bars=2, repeat_drum=True):
    '''

    :param drum_file: text file containing drum annotation
    :param loop_length_bars: number of bars per bassline
    :param repeat_drum: if True, basslines shorter than loop_length_bars will be repeated to reach the target length
    :return: drum_pattern: ['0b00010111', '0b00000111',...,'0b01000111]
    '''
    with open(drum_file, "r") as f:
        drum_txt = f.read()

    # read each midi event
    drum_at_steps = drum_txt.split("\n")[1:]
    drum_at_steps = (drum_at_step for drum_at_step in drum_at_steps if drum_at_step)  # remove empty lines
    drum_pattern = []
    for time_step, drum_at_step in enumerate(drum_at_steps):
        drum_at_step = drum_at_step.replace(",","")
        if time_step <= (loop_length_bars*16-1):
            drum_pattern.append("0b"+drum_at_step)

    number_of_drum_bands = len(drum_pattern[-1])-2 #remove 0b and count instruments

    # Create a drum annotation for a silent step  exp "0b0...0"
    silent_step = "0b"
    for i in range(number_of_drum_bands):
        silent_step+="0"

    # extend drum pattern with silence if the pattern is short of a multiple of a single bar
    while len(drum_pattern)%16!=0:
        drum_pattern.append(silent_step)

    #
    if repeat_drum:
        while len(drum_pattern)<(loop_length_bars*16):
            drum_pattern+=drum_pattern
        drum_pattern = drum_pattern[:loop_length_bars*16]

    # print("Drum File: ", drum_file," Drum Length=", len(drum_pattern))
    return drum_pattern


if __name__ == '__main__':
    prepare_train = False
    loop_length_bars = 2

    if prepare_train:
        bass_directory = "harmonic/transcription/bassline_transcription_.txt"
        drum_directory = "percussive/transcription_7Bands/drum_transcription_.txt"
        releases, bassline_files, drum_files = get_bassline_drum_text_files("../../../../dataset_soca_annotated",
                                                                            bass_directory,
                                                                            drum_directory)

        print(bassline_files[0])
        bassline_txt_to_array(bassline_files[0], translate_to=36)
        compile_basslines(releases, bassline_files, compilation_name="../data/bassline",
                          translate_to=36,
                          loop_length_bars=loop_length_bars)
        compile_drums(releases, drum_files, compilation_name="../data/drum",
                      loop_length_bars=loop_length_bars)
    else:
        bass_directory = ""
        drum_directory = "drum_transcription_.txt"
        releases, bassline_files, drum_files = get_bassline_drum_text_files("../../../../experiment_cases_Soca", bass_directory,
                                                                            drum_directory)

        compile_drums(releases, drum_files, compilation_name="../data_test/drum",
                      loop_length_bars=loop_length_bars)
