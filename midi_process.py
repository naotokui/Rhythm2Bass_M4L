#%%

import pretty_midi


# %%

import itertools 

drum_combos = []
for i in range(9 + 1):
    combos = list(itertools.combinations(range(9), i))
    combos = [sorted(c) for c in combos]
    drum_combos.extend(combos)
print(len(drum_combos), drum_combos)
# %%
