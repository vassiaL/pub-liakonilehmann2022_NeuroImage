using PyPlot
using Random
using Distributions

# Some info on graphs
function get_graph_info()
    transition_matrix_g1 =  [3 7; 5 6; 4 2; 7 1; 4 6; 1 3; 5 2]
    state_txt_g1 = ["G", "i1","s1","bG1","i2","bG2","s2"]
    stepsToGoal_g1 = [0 0; 3 2; 2 3; 4 1; 2 2; 1 3; 3 3]
    correctAction_g1 = [0, 2, 1, 2, 0, 1, 0]
    states_of_interest_g1 = [2, 3, 4, 6] # States where one action is correct and the other wrong
    # minimal_dist_g1 = [0 3 2 4 2 1 3; 0 2 3 1 2 3 3] # same as stepsToGoal_g1
    minimal_dist_g1= [0 2 2 1 2 1 3; 0 3 3 4 2 3 3]
    Vvalues_g1 = [1. 0.6769 0.6559 0.7807 0.7091 0.7952 0.6237]
    state_order_g1 = [4, 6, 2, 3, 5, 7]

    transition_matrix_g3 = [ 7 6; 1 3; 1 4; 6 2; 7 2; 5 3; 4 5]
    state_txt_g3 = ["G", "bG1","bG2","i1","i2","s1","s2"]
    stepsToGoal_g3 = [0 0; 1 2; 1 3; 3 2; 4 2; 3 2; 3 3]
    correctAction_g3 = [0, 1, 1, 2, 2, 2, 0]
    states_of_interest_g3 = [2, 3, 4, 5, 6] # States where one action is correct and the other wrong
    # minimal_dist_g3 = [0 1 1 3 4 3 3; 0 2 3 2 2 2 3] # same as stepsToGoal_g3
    minimal_dist_g3 = [0 1 1 2 2 2 3; 0 2 3 3 4 3 3]
    Vvalues_g3 = [1. 0.8641 0.8090 0.6867 0.6619 0.6619 0.6069]
    state_order_g3 = [3, 2, 5, 4, 6, 7]

    graph_info = Dict("transition_matrix_g1" => transition_matrix_g1,
                    "state_txt_g1" => state_txt_g1,
                    "stepsToGoal_g1" => stepsToGoal_g1,
                    "correctAction_g1" => correctAction_g1,
                    "states_of_interest_g1" => states_of_interest_g1,
                    "minimal_dist_g1" => minimal_dist_g1,
                    "Vvalues_g1" => Vvalues_g1,
                    "state_order_g1" => state_order_g1,
                    #
                    "transition_matrix_g3" => transition_matrix_g3,
                    "state_txt_g3" => state_txt_g3,
                    "stepsToGoal_g3" => stepsToGoal_g3,
                    "correctAction_g3" => correctAction_g3,
                    "states_of_interest_g3" => states_of_interest_g3,
                    "minimal_dist_g3" => minimal_dist_g3,
                    "Vvalues_g3" => Vvalues_g3,
                    "state_order_g3" => state_order_g3
                    )
end
function get_of_interest(graph_info, graph_id)
    if graph_id == 1
        states_of_interest = graph_info["states_of_interest_g1"]
        state_txt = graph_info["state_txt_g1"]
        stepsToGoal = graph_info["stepsToGoal_g1"]
    elseif graph_id == 3
        states_of_interest = graph_info["states_of_interest_g3"]
        state_txt = graph_info["state_txt_g3"]
        stepsToGoal = graph_info["stepsToGoal_g3"]
    end
    states_of_interest, state_txt, stepsToGoal
end
