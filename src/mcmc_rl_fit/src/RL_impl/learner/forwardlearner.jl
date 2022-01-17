"""
Forward Learner (core)
Glaescher et al. 2010

-- TSAS is initialized to Uniform (1/nr_states)
-- Q value at goal state is set to 0.
"""
mutable struct ForwardLearner
    nr_states::Int
    nr_actions::Int
    η::Float64
    γ::Float64
    initvalue::Float64
    R::Array{Float64, 1}
    QSA::Array{Float64, 2}
    TSAS::Array{Float64, 3}
    spe::Array{Float64, 1}
    do_debug_print::Bool
end
function ForwardLearner(; nr_states = 10, nr_actions = 4, η = 0.1,
    γ = .9, init_R = 0.,
    initvalue = init_R / (1. - γ),
    R = zeros(Float64, nr_states) .+ init_R,
    QSA = zeros(Float64, nr_states, nr_actions) .+ initvalue,
    TSAS = (1.0 / nr_states)*ones(nr_states, nr_actions, nr_states),
    do_debug_print = false)

    ForwardLearner(nr_states, nr_actions, η, γ, initvalue, R, QSA, TSAS,
        [0.], do_debug_print)
end
export ForwardLearner
function do_debug_print_final(learner::ForwardLearner)
    println("Final:")
    for s in 1:size(learner.QSA,1)
        @show learner.QSA[s, :]
    end
    println("--------------")
end
""" Update: main function """
function update!(learner::ForwardLearner, state, action, next_state, reward)
    learner.spe[1] = 1. - learner.TSAS[state, action, next_state]
    tsasprime = learner.TSAS[state, action, next_state] + learner.η * learner.spe[1]
    learner.TSAS[state, action, :] .*= (1. - learner.η)
    learner.TSAS[state, action, next_state] = copy(tsasprime)
    learner.R[next_state] = reward #rewards are observed. R is deterministic
    dovalueiteration!(learner)
    if abs(sum(learner.TSAS[state, action, :]) - 1.) > 0.001
        error("Tsas does not sum to 1 in row but to $(sum(TSAS[state,action,:]))")
    end
    if learner.do_debug_print
        do_debug_print_final(learner)
    end
end
export update!

function dovalueiteration!(learner::Union{ForwardLearner,ForwardLearnerBF}; maxIter = 100)

    epsilon = 0.0001 + 0.01 * maximum(learner.R) #stop iteration if max change is smaller than epsilon
    iter = 0
    maxChange = Inf
    while maxChange>epsilon && iter < maxIter
        maxChange = 0.
        for s in 1:learner.nr_states
            if (learner.R[s] > 0.) # in our paradigm, the reward state is final. Q_fwd[s,:] is undefined
                learner.QSA[s,:] .= 0.
            else
                for a in 1:learner.nr_actions
                    qsa = 0.
                    for sPrime in 1:learner.nr_states
                        qsa += learner.TSAS[s,a,sPrime] * (learner.R[sPrime] + learner.γ * maximum(learner.QSA[sPrime,:]))
                    end
                    maxChange = max(maxChange, abs(learner.QSA[s,a] - qsa))
                    learner.QSA[s,a] = copy(qsa)
                end
            end
        end
        iter +=1
    end
end
