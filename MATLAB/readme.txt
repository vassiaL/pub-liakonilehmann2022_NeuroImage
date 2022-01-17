Experiment
----------
To perform the experiment run the script start_chuv_7S_G1_LinkFlip.m (for graph 1 of paper - left column of Fig. A.1) and the script start_chuv_7S_G3_LinkFlip.m (for graph 2 of paper - right column of Fig. A.1) in MATLAB/experiment/Experiment.
Note that graph 2 = graph 3 = G3 throughout the paper and code (see namingconventions.txt).

For the experiment we used the Psychtoolbox Version 3.0.12 (Brainard, 1997).
Incompatibilities with more recent MAC OS versions might appear. We did not attempt to resolve them in order to keep the code same as when the experiment was performed.


Participants' data recording
----------------------------
As a participant performs the experiment all the events (e.g action choices) are recorded in a matrix called 'eventTimeStampTable'.
The 1st row records the timestamp of each event.
Each of the following rows corresponding to a different event. For example, the 2nd row corresponds to the onset of the interstimulus interval (ISI), the 3rd to the ID of the action selected by the participant, etc. See the script /MATLAB/experiment/Tools/getTechEnums.m for the encoding of each row.

The participant's data (eventTimeStampTable and other info) are finally stored in the structure 'taskResults'. 
The (raw) data (.mat files) of the 21 participants of this paper can be found in /MATLAB/experiment/ParticipantsData.


Processing of data
------------------
The recorded data are combined in one large eventTimeStampTable and saved in an .csv file using the script /MATLAB/experiment/DataAnalysis/PreprocessFmriFiles.m
The output file is in /MATLAB/experiment/ParticipantsData/preprocessed/SARSPEICZVG_all_fMRI.csv and in /src/mcmc_rl_fit/projects/fmri/data/SARSPEICZVG_all_fMRI.csv.
This file is then used in the julia code for model fitting, model comparison and further analysis of the behavioral data.

After the model fitting procedure the per-subject learning signals can be written back to the eventTimeStampTable of each participant (in case one wishes to perform further data analysis in MATLAB) using the script /MATLAB/experiment/DataAnalysis/add_SPE_RPE_to_EventTable.m .
The results are written in the folder MATLAB/temp/.


Simulated experiment
--------------------
To perform the experiment with simulated (Surprise Actor-critic) agents run the script runner_start_chuv_7S_G1_LinkFlip_simulation.m (for graph 1 of paper - left column of Fig. A.1) and the script runner_start_chuv_7S_G3_LinkFlip_simulation (for graph 2 of paper - right column of Fig. A.1) in MATLAB/experiment/Experiment.


Note
----
We are unfortunately not allowed to share the brain imaging data of the participants.

