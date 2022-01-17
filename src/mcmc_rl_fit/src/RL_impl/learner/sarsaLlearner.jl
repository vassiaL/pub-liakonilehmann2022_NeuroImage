"""
SARSA-λ learner (core)
"""

mutable struct SarsaL
    nr_states::Int
    nr_actions::Int
    α::Float64
    α_eff::Float64
    γ::Float64
    λ::Float64
    initvalue::Float64
    QSA::Array{Float64, 2}
    eligTrace::Array{Float64, 2}
    rpe::Array{Float64, 1}
    use_replacing_trace::Bool
    do_debug_print::Bool
end
function SarsaL(; nr_states = 10, nr_actions = 4, α = 0.1, γ = .9,
    λ = 0.9, initvalue = 0.,
    QSA = zeros(Float64, nr_states, nr_actions) .+ initvalue,
    use_replacing_trace = true,
    do_debug_print = false)

    SarsaL(nr_states, nr_actions, α, α, γ, λ, initvalue, QSA,
        zeros(Float64, nr_states, nr_actions), [0.],
        use_replacing_trace, do_debug_print)
end
export SarsaL
function do_debug_print_final(learner::Union{SarsaL, SarsaLModulated})
    println("Final:")
    for s in 1:size(learner.QSA,1)
        @show learner.QSA[s, :]
    end
    println("--------------")
end

""" Update: main function """
function update!(learner::Union{SarsaL, SarsaLModulated}, state, action, next_state, next_action, reward,
                isendofexperiment, modulatingfactor)

    if learner.use_replacing_trace # replacing trace
        learner.eligTrace[state,action] = 1.
    else # (accumulating) eligibility trace
        learner.eligTrace[state,action] += 1.
    end

    getlearningrate!(learner, modulatingfactor)

    if reward>0 #goal state, reward, NO NEXT ACTION (in our paradigm)
        learner.rpe[1] = (reward - learner.QSA[state, action])
        learner.QSA[:, :] .+= learner.α_eff * learner.eligTrace * learner.rpe[1] #e_trace for the current state is 1
        learner.eligTrace = zeros(Float64, learner.nr_states, learner.nr_actions) # reset ET at end of episode
    else #non goal state
        if !isendofexperiment
            learner.rpe[1] = reward + learner.γ * learner.QSA[next_state, next_action] - learner.QSA[state, action]
            learner.QSA[:, :] .+= learner.α_eff * learner.eligTrace * learner.rpe[1]
            learner.eligTrace[:, :] .*= learner.λ * learner.γ
        else
            learner.rpe[1] = 0. # end of experiment
        end
    end
    # @show state, action, next_state, reward
    # @show learner.rpe[1]
    # @show learner.eligTrace[state, 1], learner.eligTrace[state, 2]
    # @show learner.QSA[state, 1], learner.QSA[state, 2]
    # @show learner.QSA
    if learner.do_debug_print
        do_debug_print_final(learner)
    end
end
export update!
function getlearningrate!(learner::Union{SarsaL, REINFORCE, REINFORCEbaseline}, modulatingfactor)
    learner.α_eff = learner.α
end
