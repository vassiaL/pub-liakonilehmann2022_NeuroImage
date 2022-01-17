using Random
"""
Prioritized Sweeping (core)

Implementation of Algorithm 2 Prioritized Sweeping with reversed full backups in
[Van Seijen, Harm, and Richard S. Sutton. "Efficient planning in MDPs by small backups."
Proceedings of the 30th International Conference on International Conference on Machine Learning. Vol. 28. 2013]

Note 1
------
The Tsas' estimation is NOT done through Perfect Integration,
but through the estimators :
TParticleFilterJump (particle filtering derived for an outlier detection task)

The Rsas' estimation is done with Perfect Integration.

The fact the estimations for Tsas' and Rsas' are different prevents us from using the
    Small Backups method of [Van Seijen and Sutton, 2013].
We use reverse full backups instead.

Note 2
------
Perfect Integration with Small Backups is not supported right now.
"""

mutable struct PrioritizedSweeping{TREstimate,TTEstimate}
    nr_states::Int
    nr_actions::Int
    γ::Float64
    init_R::Float64
    initvalue::Float64
    nr_upd_cycles::Int
    counter::Int
    QSA::Array{Float64, 2}
    VS::Array{Float64, 1}
    US::Array{Float64, 1}
    Restimate::TREstimate
    Testimate::TTEstimate
    prioQ::PriorityQueue
    do_debug_print::Bool
end
function PrioritizedSweeping(; nr_states = 10, nr_actions = 4, γ = .9, init_R = 1.,
    initvalue = init_R / (1. - γ),
    nr_upd_cycles = 3, counter = 0,
    QSA = zeros(Float64, nr_states, nr_actions) .+ initvalue,
    VS = zeros(Float64, nr_states) .+ initvalue,
    US = zeros(Float64, nr_states) .+ initvalue,
    Restimatetype = RIntegratorNextStateReward,
    Testimatetype = TParticleFilterJump,
    prioQ = PriorityQueue(Base.Order.Reverse, zip(Int[], Float64[])),
    nparticles = 3, surpriseprobability = .01, stochasticity = .01, seedlearner = 3,
    etaleak = .9,
    do_debug_print = false)

    if Testimatetype == TParticleFilterJump
        Testimate = TParticleFilterJump(nr_states = nr_states,
                                nr_actions = nr_actions,
                                nparticles = nparticles,
                                surpriseprobability = surpriseprobability,
                                stochasticity = stochasticity,
                                seed = seedlearner)
    elseif Testimatetype == TLeakyIntegrator
        Testimate = TLeakyIntegrator(nr_states = nr_states,
                                        nr_actions = nr_actions,
                                        etaleak = etaleak)
    elseif Testimatetype == TLeakyGlaescher
        Testimate = TLeakyGlaescher(nr_states = nr_states,
                                    nr_actions = nr_actions,
                                    η = etaleak)
    end
    # ---- Set up Rsas' estimator ----
    if Restimatetype == RIntegratorNextStateReward
        Restimate = RIntegratorNextStateReward(nr_states = nr_states, init_R = init_R)
    end

    PrioritizedSweeping(nr_states, nr_actions, γ, init_R, initvalue, nr_upd_cycles, counter,
                QSA, VS, US, Restimate, Testimate, prioQ, do_debug_print)
end
export PrioritizedSweeping
""" Add state and priority value to queue """
function addtoqueue!(prioQ, state, prio)
    if haskey(prioQ, state)
        prioQ[state] = prio
    else
        enqueue!(prioQ, state, prio)
    end
    prioQ
end
""" Go through queue """
function processqueue!(learner::PrioritizedSweeping)
    while length(learner.prioQ) > 0 && learner.counter < learner.nr_upd_cycles
        learner.counter += 1
        sprime = dequeue!(learner.prioQ)
        ΔV = learner.VS[sprime] - learner.US[sprime]
        learner.US[sprime] = learner.VS[sprime]
        updatepredecessorsq!(learner, sprime, ΔV)
    end
    learner.counter = 0
end
""" Update Q values of predecessors - For Testimate that uses counts (Ns1a0s0) """
function updatepredecessorsq!(learner::PrioritizedSweeping{TR, TT} where {TR, TT<:TNs1a0s0},
                            sprime, ΔV)
    if length(learner.Testimate.Ns1a0s0[sprime]) > 0
        for ((a0, s0), n) in learner.Testimate.Ns1a0s0[sprime]
            if n > 0.
                learner.QSA[s0, a0] += learner.γ * ΔV * n/learner.Testimate.Nsa[a0, s0]
                updateV!(learner, s0)
            end
        end
    end
end
""" Update Q values of predecessors - For Testimate that uses probabilites (Ps1a0s0) """
function updatepredecessorsq!(learner::PrioritizedSweeping{TR, TT} where {TR, TT<:TPs1a0s0},
                            sprime, ΔV)
    if length(learner.Testimate.Ps1a0s0[sprime]) > 0
        for ((a0, s0), n) in learner.Testimate.Ps1a0s0[sprime]
            if learner.Testimate.Ps1a0s0[sprime][a0, s0] > 0.
                learner.QSA[s0, a0] += learner.γ * ΔV * n
                updateV!(learner, s0)
            end
        end
    end
end

""" Full backup  -- R[s']"""
function updateq!(learner::PrioritizedSweeping{RIntegratorNextStateReward, TT} where TT,
                state, action, next_state, reward, done)
    # Update all s-a space (It is less efficient and it won't make much difference,
    # but we do it like this, because it is the correct way: The reason is that
    # we have R(s') and not R(s,a), as they had in the van Seijen paper!)
    shuffledstates = shuffle(collect(1:learner.nr_states))
    for s in shuffledstates
        for a in 1:learner.nr_actions
            next_values, next_transprobs, next_states = getnextstates(learner, s, a)
            Rbar = [learner.Restimate.R[sprime] for sprime in next_states]
            learner.QSA[s, a] = sum(next_transprobs.* (Rbar + learner.γ * next_values))
        end
        updateV!(learner, s)
    end
end
""" Get next states, trans_probs to next states, and values of next states
    -- For Testimate that uses probabilities (Ps1a0s0) """
function getnextstates(learner::PrioritizedSweeping{TR, TT} where {TR, TT<:TPs1a0s0}, state, action)
    next_states = [s for s in 1:learner.nr_states if haskey(learner.Testimate.Ps1a0s0[s],(action, state))]
    next_values = [learner.US[s] for s in next_states]
    # @show nextvs
    next_transprobs = [learner.Testimate.Ps1a0s0[s][action, state] for s in next_states]
    # @show nextps
    next_values, next_transprobs, next_states
end
""" Get next states, trans_probs to next states, and values of next states
    -- For Testimate that uses counts (Ns1a0s0)
    NOT USED IN PAPER
    """
function getnextstates(learner::PrioritizedSweeping{TR, TT} where {TR, TT<:TNs1a0s0}, state, action)
    next_states = [s for s in 1:learner.nr_states if haskey(learner.Testimate.Ns1a0s0[s],(action, state))]
    next_values = [learner.US[s] for s in next_states]
    # @show nextvs
    next_transprobs = [learner.Testimate.Ns1a0s0[s][action, state]/learner.Testimate.Nsa[action, state] for s in next_states]
    # @show nextps
    next_values, next_transprobs, next_states
end
""" Update V value and add to queue """
function updateV!(learner::PrioritizedSweeping, state)
    learner.VS[state] = maximum(learner.QSA[state, :])
    prio = abs(learner.VS[state] - learner.US[state])
    # @show learner.VS[s], learner.US[s], p
    addtoqueue!(learner.prioQ, state, prio)
    # @show learner.prioQ
    if learner.do_debug_print
        println("prio=$(round(prio,digits=3))")
    end
end

function do_debug_print_beforeupdate(learner::PrioritizedSweeping,
                                    state, action, done)
    if done
        println(" **************** DONE!!! ************** ")
    end
    println("Before update:")
    if in(:Ps1a0s0, fieldnames(typeof(learner.Testimate)))
        for s in 1:learner.nr_states
            @show learner.Testimate.Ps1a0s0[s][(action, state)]
        end
    else
        for s in 1:learner.nr_states
            @show learner.Testimate.Ns1a0s0[s][(action, state)]
        end
    @show learner.Testimate.Nsa[action, state]
    end
    @show learner.Restimate.R
    for s in 1:size(learner.QSA, 1)
        @show learner.QSA[s, :]
    end
end
function do_debug_print_afterupdate(learner::PrioritizedSweeping,
                                state, action)
    println("After update:")
    if in(:Ps1a0s0, fieldnames(typeof(learner.Testimate)))
        for s in 1:learner.nr_states
            @show learner.Testimate.Ps1a0s0[s][(action, state)]
        end
    else
        for s in 1:learner.nr_states
            @show learner.Testimate.Ns1a0s0[s][(action, state)]
        end
        @show learner.Testimate.Nsa[action, state]
    end
    @show learner.Restimate.R
    for s in 1:size(learner.QSA, 1)
        @show learner.QSA[s, :]
    end
end
function do_debug_print_final(learner::PrioritizedSweeping)
    println("Final:")
    for s in 1:size(learner.QSA,1)
        @show learner.QSA[s, :]
    end
    println("--------------")
end

""" Update: main function """
function update!(learner::PrioritizedSweeping, state, action, next_state, reward, done)

    if learner.do_debug_print
        do_debug_print_beforeupdate(learner, state, action, done)
    end

    # Update Tsas'
    updatet!(learner.Testimate, state, action, next_state, done)
    # Update Rsas'
    updater!(learner.Restimate, state, action, next_state, reward)
    # Update Q values
    updateq!(learner, state, action, next_state, reward, done)

    if learner.do_debug_print
        do_debug_print_afterupdate(learner, state, action)
    end

    # Go through priotity queue
    processqueue!(learner)

    if learner.do_debug_print
        do_debug_print_final(learner)
    end
end
export update!
