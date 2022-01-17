# Helper functions

function get_all_puzzles(d::Array{Int,2})
    all_puzzles = d[:,Int(RL_Fit.c_puzzle)]
    sort(unique(all_puzzles))
end
function get_all_episodes_in_puzzle(d::Array{Int,2}, i_puzzle::Int)
    idx = d[:,Int(RL_Fit.c_puzzle)].== i_puzzle
    all_epi = d[idx,Int(RL_Fit.c_episode)]
    sort(unique(all_epi))
end
function get_data_by_puzzle_episode(d::Array{Int,2}, i_puzzle::Int, i_episode::Int)
    idx = (d[:,Int(RL_Fit.c_puzzle)].== i_puzzle) .& (d[:,Int(RL_Fit.c_episode)].==i_episode)
    d[idx,:]
end
function get_data_stats_and_dimensions(data)
    data_stats = get_data_stats(data)
    nr_states = data_stats.max_state_id
    nr_actions = data_stats.max_action_id
    (nr_samples,) = size(data)
    data_stats, nr_states, nr_actions, nr_samples
end

converttoInt(variable) = typeof(variable) == Float64 ? Int(round(variable)) : variable
