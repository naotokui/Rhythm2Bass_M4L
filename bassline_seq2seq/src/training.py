#%%

import seq2seq_LSTM
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG

%pylab inline

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
        print(k, b[0].shape)

# %%
