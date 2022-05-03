Data for the emulation of spatial-temporal fields.
Each data should contain 3 fidelity fields. Each has 512 training and 128 testing points.
If you would like to do a two-fidelity problem, you can choose the 1st and the 3rd fidelity data as your low- and high-fidelity data.

Data formate:
xtr: train data input. N*d matrix
Ytr: train data output. Each cell corresponds to a fidelity output collection (the lower number of cell is for lower fidelity). In each cell, the data is an N*d1*d2 tensor; each d1*d2 can be considered an image. Because of the fidelity, outputs from different fidelity have different sizes (d1 and d2).
Ytr_interp: interpolation based on Ytr such that the sizes are the same across different fidelities.

xte: test data input
Yte: test data output. Same formate as Ytr.
Yte_interp: test data output interpolations. Same formate as Ytr.

lvSetting: information for fidelity setting
logg: log files

