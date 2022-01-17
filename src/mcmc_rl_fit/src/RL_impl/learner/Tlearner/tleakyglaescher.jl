"""
TLeakyGlaescher: delta-rule updates of the transition matrix
as in Glaescher et al., 2010
"""
struct TLeakyGlaescher <: TPs1a0s0
    nr_states::Int
    nr_actions::Int
    η::Float64
    SPE::Array{Float64, 1}
    Ps1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    terminalstates::Array{Int,1}
end
function TLeakyGlaescher(; nr_states = 10, nr_actions = 4, η = .9)
    SPE = [0.]
    Ps1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:nr_states]
    [Ps1a0s0[sprime][(a, s)] = 1. /nr_states for sprime in 1:nr_states for a in 1:nr_actions for s in 1:nr_states]
    TLeakyGlaescher(nr_states, nr_actions, η, SPE, Ps1a0s0, Int[])
end
export TLeakyGlaescher
""" X[t] = etaleak * X[t-1] + etaleak * I[.] """
function updatet!(learnerT::TLeakyGlaescher, s0, a0, s1, done)
    calcSPE!(learnerT, s0, a0, s1)
    leaka0s0!(learnerT, s0, a0, s1, done)
    computeterminalPs1a0s0!(learnerT, s1, done)
end
export updatet!
function leaka0s0!(learnerT::TLeakyGlaescher, s0, a0, s1, done)
    tsasprime = learnerT.Ps1a0s0[s1][(a0, s0)] + learnerT.η * learnerT.SPE[1]
    for sprime in 1:learnerT.nr_states
        learnerT.Ps1a0s0[sprime][(a0, s0)] *= (1. - learnerT.η)
    end
    learnerT.Ps1a0s0[s1][(a0, s0)] = copy(tsasprime)
end
function calcSPE!(learnerT::TLeakyGlaescher, s0, a0, s1)
    learnerT.SPE[1] = 1. - learnerT.Ps1a0s0[s1][(a0, s0)]
end
