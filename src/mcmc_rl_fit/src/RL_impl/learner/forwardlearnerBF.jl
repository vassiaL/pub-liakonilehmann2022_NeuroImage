"""
Forward Learner (ie Value iteration) with Bayes-factor mediated updates
(via particle filtering)
core

-- TSAS is initialized to Uniform (1/nr_states)
-- Q value at goal state is set to 0.
"""
mutable struct ForwardLearnerBF{TTEstimate}
    nr_states::Int
    nr_actions::Int
    γ::Float64
    initvalue::Float64
    R::Array{Float64, 1}
    QSA::Array{Float64, 2}
    TSAS::Array{Float64, 3}
    Testimate::TTEstimate
    do_debug_print::Bool
end
function ForwardLearnerBF(; nr_states = 10, nr_actions = 4,
    γ = .9, init_R = 0.,
    initvalue = init_R / (1. - γ),
    R = zeros(Float64, nr_states) .+ init_R,
    QSA = zeros(Float64, nr_states, nr_actions) .+ initvalue,
    TSAS = (1.0 / nr_states)*ones(nr_states, nr_actions, nr_states),
    do_debug_print = false,
    nparticles = 3, surpriseprobability = .01, stochasticity = .01, seedlearner = 3,)

    Testimate = TParticleFilterJump(nr_states = nr_states,
                            nr_actions = nr_actions,
                            nparticles = nparticles,
                            surpriseprobability = surpriseprobability,
                            stochasticity = stochasticity,
                            seed = seedlearner)

    ForwardLearnerBF(nr_states, nr_actions, γ, initvalue, R, QSA, TSAS,
        Testimate, do_debug_print)
end
export ForwardLearnerBF

""" Update: main function """
function update!(learner::ForwardLearnerBF, state, action, next_state, reward)

    updatet!(learner.Testimate, state, action, next_state, reward>0)

    for sPrime in 1:learner.nr_states
        learner.TSAS[state, action, sPrime] = copy(learner.Testimate.Ps1a0s0[sPrime][(action, state)])
    end

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
