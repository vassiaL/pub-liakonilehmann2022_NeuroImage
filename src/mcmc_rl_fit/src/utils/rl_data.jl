# Pre- and post- processing of behavioral and simulated data.
# file IO
using DelimitedFiles

function log_print!(buffer::IOBuffer, line::String)
    print(line)
    print(buffer, line)
end

function log_print_ln!(buffer::IOBuffer, line::String)
    println(line)
    println(buffer, line)
end

@enum SARSPE_COLS c_state=1 c_action=2 c_reward=3 c_nextState=4 c_puzzle=5 c_episode=6 c_subject=7 c_surprise_trial=8 c_RT_prctile=9 c_action_value=10 c_graph_version=11


function filter_by_person(data::Array{Int,2}, person_id::Int)
    idx = data[:,Int(c_subject)] .== person_id
    data[idx,:]
end

function filter_by_person(data::Array{Int,2}, person_ids::Set)
    filtered_data = Array{Int, 2}(undef, 0, size(data)[2])
    for id in person_ids
        subj_data = filter_by_person(data, id)
        filtered_data = vcat(filtered_data, subj_data)
    end
    filtered_data
end

function filter_by_max_episode(data, max_episode_id)
    idx = data[:,Int(c_episode)].<=max_episode_id
    data[idx,:]
end

function filter_by_episode(data, episode_id)
    idx = data[:,Int(c_episode)].==episode_id
    data[idx,:]
end


function filter_by_max_action(data, max_nr_actions_per_subj)
    allSubj = sort(unique(data[:, Int(c_subject)]))
    (m,n) = size(data)
    allData = zeros(Int,(0,n))
    allSubj = sort(unique(data[:, Int(c_subject)]))
    for subj in allSubj
      data_subj = data[data[:,Int(c_subject)].==subj,:]
      (m,n) = size(data_subj)
      maxIdx = min(m, max_nr_actions_per_subj)
      data_subj = data_subj[1:maxIdx,:]
      allData = vcat(allData, data_subj)
    end
    allData
end

function load_SARSPEI(data_path; subject_id::Int = 0, max_episodes_per_subj::Int = typemax(Int), max_nr_actions_per_subj::Int = typemax(Int), sub_id_offset = 0)
    # sub_id_offset is used to concat multiple files where subject ID-n would reappaear
    data = collect(transpose(readdlm(data_path, ',', Int)))
    if sub_id_offset>0
        println("Warning: subject IDs offset by $sub_id_offset")
        data[:,Int(RL_Fit.c_subject)] += sub_id_offset
    end

    if subject_id != 0
        data=filter_by_person(data,subject_id)
    end
    data = filter_by_max_action(data, max_nr_actions_per_subj)
    data = filter_by_max_episode(data, max_episodes_per_subj)
    reward_rows = data[:,Int(c_reward)] .!=0 # use unit reward
    data[reward_rows,Int(c_reward)] .= 1
    stats = get_data_stats(data)
    data, stats
end

struct Data_stats
    nr_subjects::Int
    all_subject_ids::Array{Int,1}
    nr_puzzles::Int
    nr_episodes::Int
    nr_actions::Int
    max_state_id::Int
    max_action_id::Int
    nr_rewards::Int
end

function to_string(data_stats::Data_stats)
    #pretty print for console
    txt = string("nr_subjects: $(data_stats.nr_subjects)\n",
    "all_subject_ids: $(data_stats.all_subject_ids)\n",
    "nr_puzzles: $(data_stats.nr_puzzles)\n",
    "nr_episodes: $(data_stats.nr_episodes)\n",
    "nr_actions: $(data_stats.nr_actions)\n",
    "max_state_id: $(data_stats.max_state_id)\n",
    "max_action_id: $(data_stats.max_action_id)")
end

function to_short_string(data_stats::Data_stats)
    "S:$(data_stats.nr_subjects)|P:$(data_stats.nr_puzzles)|E:$(data_stats.nr_episodes)|A:$(data_stats.nr_actions)"
end


function get_data_stats(data::Array{Int,2})
    allSubj = sort(unique(data[:, Int(c_subject)]))
    nr_subjects = length(allSubj)
    nr_actions = size(data)[1]
    #count all episodes per subject
    nr_puzzles = 0
    nr_episodes = 0
    for s = allSubj
        idx = data[:,Int(c_subject)].== s
        nr_puzzles += length(unique(data[idx,Int(c_puzzle)]))
        nr_episodes += length(unique(data[idx,Int(c_episode)]))
    end
    #nr of different states and actions
    diff_states = unique( vcat( data[:,Int(c_state)], data[:,Int(c_nextState)] ))

    max_state_id = maximum(diff_states)
    max_action_id = maximum(data[:,Int(c_action)])
    nr_rewards = sum(data[:,Int(c_reward)] .!= 0)
    stats = Data_stats(nr_subjects, allSubj, nr_puzzles, nr_episodes, nr_actions, max_state_id, max_action_id, nr_rewards)
end

struct Step_by_step_signals
    subject_id::Int
    RPE::Array{Float64,1}
    SPE::Array{Float64,1}
    action_selection_prob::Array{Float64,1}
    LL::Float64
    data_stats::Data_stats
    algorithm_name::String
    param_vector::Array{Float64,1}
    param_string::String
    max_bias_action_index::Int
    results_dict::Dict{String,Any}  # optionally add more algo-specific results
end


function to_string(signals::Step_by_step_signals; round_nr_digits=3)
    txt = string("subject_id: $(signals.subject_id)\n",
      "algorithm_name: $(signals.algorithm_name)\n",
      "param_string: $(signals.param_string)\n",
      "param_vector: $(signals.param_vector)\n",
      "LL: $(round(signals.LL,digits=round_nr_digits))\n",
      "RPE\n$(round.(signals.RPE,digits=round_nr_digits))\n",
      "SPE\n$(round.(signals.SPE,digits=round_nr_digits))\n",
      "action_selection_prob\n$(round.(signals.action_selection_prob,digits=round_nr_digits))\n",
      "results_dict\n$(signals.results_dict)")
end
