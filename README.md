# Emotion-Style Transfer for Music
## Objective: Develop and train a model which can combine the *emotion* of one song with the *content* of another, in a manner similar to neural style transfer.

### *NOTE: The repo currently does not contain the data because the folder is too large.*
* In my local repo, there is a `data` folder which contains two subfolders: `MER_taffc_as_wav` and `MER_taffc_standard`.
* The `MER_taffc_as_wav` folder contains the audio files from the MER TAFFC dataset in WAV format.
* The `MER_taffc_standard` folder contains the songs from the MER TAFFC dataset, split up into ~5 second segments.
* In both subfolders, the audio files are divided into folders `Q1`, `Q2`, `Q3`, and `Q4` according to which quadrant of the valence-arousal plane they correspond to.

### Repo files:
* `data_transformation.ipynb`: Explore and transform data to put it in the appropriate format. Specifically, files are normalized so that the audio amplitudes of each file fall in the range $[-1,1]$, and the files are split into segments of $111,132$ samples and saved as WAV files.
* `mer_model.ipynb`: Transform audio segments into log mel-spectrograms, split the segments into stratified train and test sets, build and compile the model, and train the model.
* `mer_model_1_25_23.h5` and `mer_model_1_26_23.h5`: Trained Keras MER models.
