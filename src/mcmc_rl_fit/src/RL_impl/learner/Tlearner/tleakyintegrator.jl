"""
TLeakyIntegrator: Integration with a constant leak
(No leakage of background non-currently experienced s-a pairs)

NOT USED IN PAPER!
"""
struct TLeakyIntegrator <: TNs1a0s0
    nr_states::Int
    nr_actions::Int
    etaleak::Float64
    lowerbound::Float64
    SPE::Array{Float64, 1}
    Nsa::Array{Float64, 2}
    Ns1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    terminalstates::Array{Int,1}
end
function TLeakyIntegrator(; nr_states = 10, nr_actions = 4, etaleak = .9, lowerbound = eps())
    SPE = [0.]
    Nsa = zeros(nr_actions, nr_states) .+ nr_states*eps()
    Ns1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:nr_states]
    [Ns1a0s0[sprime][(a, s)] = eps() for sprime in 1:nr_states for a in 1:nr_actions for s in 1:nr_states]
    TLeakyIntegrator(nr_states, nr_actions, etaleak, lowerbound, SPE, Nsa, Ns1a0s0, Int[])
end
export TLeakyIntegrator
""" X[t] = etaleak * X[t-1] + etaleak * I[.] """
function updatet!(learnerT::TLeakyIntegrator, s0, a0, s1, done)
    calcSPE!(learnerT, s0, a0, s1)
    leaka0s0!(learnerT, s0, a0, s1, done)
    computeterminalNs1a0s0!(learnerT, s1, done)
end
export updatet!
function leaka0s0!(learnerT::TLeakyIntegrator, s0, a0, s1, done)
    learnerT.Nsa[a0, s0] *= learnerT.etaleak # Discount transition
    learnerT.Nsa[a0, s0] += learnerT.etaleak # Increase observed transition
    learnerT.Nsa[a0, s0] += (1. - learnerT.etaleak) * learnerT.nr_states * learnerT.lowerbound # Bound it

    nextstates = [s for s in 1:learnerT.nr_states if haskey(learnerT.Ns1a0s0[s],(a0,s0))]
    for sprime in nextstates
        learnerT.Ns1a0s0[sprime][(a0, s0)] *= learnerT.etaleak # Discount all outgoing transitions
        learnerT.Ns1a0s0[sprime][(a0, s0)] += (1. - learnerT.etaleak) * learnerT.lowerbound # Bound it
    end
    if haskey(learnerT.Ns1a0s0[s1], (a0, s0))
        learnerT.Ns1a0s0[s1][(a0, s0)] += learnerT.etaleak # Increase observed
    else

        learnerT.Ns1a0s0[s1][(a0, s0)] = learnerT.etaleak
        learnerT.Ns1a0s0[s1][(a0, s0)] += (1. - learnerT.etaleak) * learnerT.lowerbound # Bound it
    end
end
function computeterminalNs1a0s0!(learnerT::TLeakyIntegrator, s1, done)
    if done
        if !in(s1, learnerT.terminalstates)
            push!(learnerT.terminalstates, s1)
            for a in 1:learnerT.nr_actions
                for s in 1:learnerT.nr_states
                    if s == s1
                        learnerT.Nsa[a, s1] = 1.
                        learnerT.Ns1a0s0[s][(a, s1)] = 1.
                    else
                        learnerT.Ns1a0s0[s][(a, s1)] = 0.
                    end
                end
            end
        end
    end
end
function calcSPE!(learnerT::TLeakyIntegrator, s0, a0, s1)
    Psasprime = 0.
    if learnerT.Nsa[a0, s0]!=0
        if haskey(learnerT.Ns1a0s0[s1], (a0, s0))
            Psasprime = learnerT.Ns1a0s0[s1][(a0, s0)]/learnerT.Nsa[a0, s0]
        end
    end
    learnerT.SPE[1] = 1. - Psasprime
end
