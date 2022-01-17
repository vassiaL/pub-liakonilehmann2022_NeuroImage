function add_SPE_RPE_to_EventTable(algoname)
% writes the signals (RPE, SPE, etc) resulting from the per-subject Metropolis 
% fit into the EventTimestampTable.
% 
% algoname = name of folder in '../../../src/mcmc_rl_fit/projects/fmri/someresults/fits/'
%
% For example: 
algoname = 'actor_critic_TD_modulatedActor_1degree';

%% Path of participants; data stuff
data_path = '../';

%% Path of Model fit results
fit_path = '../../../src/mcmc_rl_fit/projects/fmri/someresults/fits/';

%% Where to save the mat files
save_path = '../../temp';
save_path = fullfile(save_path, strcat(algoname, '_resultmatFiles'));
mkdir(save_path);
%% Load subj ID mapping
load(fullfile(data_path, 'ParticipantsData/preprocessed/userId_FileMap.mat'));
%%
fitting_results_folder = fullfile(fit_path, algoname,'/maxLL_fit/');

for iSubj = 1:length(userId_FileMap)
%for iSubj = 1:2
    subj_file = userId_FileMap{iSubj};
    taskResultFile = [fullfile(data_path,'ParticipantsData/'), subj_file];
    load(taskResultFile);
    taskResults.TaskDefinition.setUp.subjId   
    eventTbl = taskResults.eventTimeStampTable;
    idx_actions = find(taskResults.eventTimeStampTable(4,:)>0);
    idx_next_state = idx_actions(:)+2; %we place the SPE (and RPE) at the OUTCOME of the action (=NEXT_STATE)
    idx_states = find(taskResults.eventTimeStampTable(6,:)>0); %we place the Values at the states (cause also value at start states - ie no action before)
   
    % load the fitting results:
    %% RPE
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'RPE', 'pop', eventTbl, idx_next_state);
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'RPE', 'subj', eventTbl, idx_next_state);
    %% SPE
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'SPE', 'pop', eventTbl, idx_next_state);
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'SPE', 'subj', eventTbl, idx_next_state);
    %% gamma_surprise
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'gamma_surprise', 'pop', eventTbl, idx_next_state);
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'gamma_surprise', 'subj', eventTbl, idx_next_state);
    %% action_selection_prob
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'action_selection_prob', 'pop', eventTbl, idx_actions);
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'action_selection_prob', 'subj', eventTbl, idx_actions);
    %% Sbf
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'Surprise', 'pop', eventTbl, idx_next_state);
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'Surprise', 'subj', eventTbl, idx_next_state);
    %% Delta policy parameters
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'DeltaPolicy', 'pop', eventTbl, idx_next_state);
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'DeltaPolicy', 'subj', eventTbl, idx_next_state);
    %% Delta policy parameters
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'StateValue', 'pop', eventTbl, idx_states);
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'StateValue', 'subj', eventTbl, idx_states);
    %% PolicyParametersDiffChosenVsUnchosen (p(s,a_chosen) - p(s,a_notchosen))
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'PolicyParametersDiffChosenVsUnchosen', 'pop', eventTbl, idx_actions);
    eventTbl = getSignalFromFile(fitting_results_folder, iSubj, 'PolicyParametersDiffChosenVsUnchosen', 'subj', eventTbl, idx_actions);
    
    %% Pass new eventTimeStampTable
    taskResults.eventTimeStampTable = eventTbl;
   
    %% write the params and the fitting protocol to the taskResults
    fitting_params = struct();

    fitting_params.algorithmname = algoname;
   
    fitting_params.pop_params  = transpose(csvread([fitting_results_folder , 'subj_', num2str(iSubj), '/params_pop.csv']));
    fitting_params.subj_params  = transpose(csvread([fitting_results_folder , 'subj_', num2str(iSubj), '/params_subj.csv']));
    fitting_params.protocol = fileread([fitting_results_folder , 'subj_', num2str(iSubj), '/description.txt']);
    
    taskResults.fitting_params = fitting_params;
    
    res_file  = userId_FileMap{iSubj};
    res_file =  [res_file(1:end-4), '_', algoname, '_signalsMCMCfit.mat'];
    save(fullfile(save_path, res_file), 'taskResults')
   
end

end
function eventTbl = getSignalFromFile(fitting_results_folder, iSubj, signalname, fitType, eventTbl, idx_inEventTBl)
    
    fit_res_file = [fitting_results_folder , 'subj_', num2str(iSubj), '/', signalname, '_', fitType, '.csv'];
    
    rownumber = whichresultRow(signalname, fitType);
    if exist(fit_res_file, 'file')
        signal_fitType = transpose(csvread(fit_res_file));
        
        if strcmp(signalname, 'StateValue') && length(idx_inEventTBl) > length(signal_fitType(1,:)) % There is an extra state before ESC was pressed
            signal_fitType = [signal_fitType, zeros(1, (length(idx_inEventTBl) - length(signal_fitType)))];
        end
        eventTbl(rownumber, idx_inEventTBl) = signal_fitType(1,:);
    else
        eventTbl(rownumber, :) = zeros(1, size(eventTbl,2));
    end 
end










