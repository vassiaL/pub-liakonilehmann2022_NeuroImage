This is the code for the publication:

V. Liakoni *, M. Lehmann *, A. Modirshanechi, J. Brea, A. Lutti, W. Gerstner ** , K. Preuschoff **
[*Brain signals of a Surprise-Actor-Critic model: Evidence for multiple learning modules in human decision making*](https://doi.org/10.1016/j.neuroimage.2021.118780), NeuroImage 246, 1053-8119 (2022)

\* V.L. and M.L. made equal contribution to this article. \
\** W.G. and K.P. made equal contribution to this article.

Contact:
[vasiliki.liakoni@gmail.com](mailto:vasiliki.liakoni@gmail.com), [marco.lehmann@gmail.com](mailto:marco.lehmann@gmail.com)

## Installation

Dependencies:

* Mac or Linux
* [Julia](https://julialang.org) (1.2)
* MATLAB R2015a

### For the julia code

Navigate into the src folder. \
Open a julia terminal, press "]" to enter the package management mode and type

(v1.2) pkg> activate .. \
(SinergiafMRI_datafit) pkg> instantiate

All (julia) packages and dependencies will be installed automatically within this environment.

### For the MATLAB code

Navigate into the MATLAB folder and add all subfolders to your MATLAB path.

## Usage

### For the julia code

julia> using SinergiafMRI_datafit

To run the model fitting and model comparison procedures of the paper, type

julia> SinergiafMRI_datafit.runner_crossval_multipletimes_fit()

Note that these procedures would take long, so the settings are different from the ones used in the paper. \
Uncomment the marked lines in the fmri_run_multipletimes.jl file to perform the analysis using the  paper's settings.

A good place to start playing with the data is the simpleFittingTest.jl file.

For more information, please see the src/readme.txt.

### For the MATLAB code

To run the experiment run the script start_chuv_7S_G1_LinkFlip.m (for graph 1 of paper - left column of Fig. A.1) and the script start_chuv_7S_G3_LinkFlip.m (for graph 2 of paper - right column of Fig. A.1).

For more information, please see the MATLAB/readme.txt.

## Code

* MATLAB/experiment: the paper's experiment, participants' data recording and some data preprocessing.
* MATLAB/recovery: the paper's experiment (same as above, but without visualizations) with simulated Surprise Actor-critic agents.
* src/mcmc_rl_fit/src: RL algorithms, model fitting and model comparison (via crossvalidation).
* src/mcmc_rl_fit/fmri: high level runners, data analysis, plotting.
Please refer to src/readme.txt and to MATLAB/readme.txt for more information.


## Data and Figures

* /MATLAB/experiment/ParticipantsData: the participants's (raw) data.
* /src/mcmc_rl_fit/projects/fmri/data: all participants' data concatenated. The file SARSPEICZVG_all_fMRI.csv is used for all analyses within the julia code.
* /src/mcmc_rl_fit/projects/fmri/data_sim: all simulated agents' data concatenated.
* data: final results used for the paper's figures.
* figs: latex source code to reproduce the paper's figures.
