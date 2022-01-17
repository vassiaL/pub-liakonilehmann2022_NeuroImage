"""
SARSA-λ learner with surprise-modulated learning rate (core)
"""

mutable struct SarsaLModulated
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
function SarsaLModulated(; nr_states = 10, nr_actions = 4, α = 0.1, γ = .9,
    λ = 0.9, initvalue = 0.,
    QSA = zeros(Float64, nr_states, nr_actions) .+ initvalue,
    use_replacing_trace = true,
    do_debug_print = false)

    SarsaLModulated(nr_states, nr_actions, α, α, γ, λ, initvalue, QSA,
        zeros(Float64, nr_states, nr_actions), [0.],
        use_replacing_trace, do_debug_print)
end
export SarsaLModulated
function getlearningrate!(learner::Union{SarsaLModulated, REINFORCEModulated},
                        modulatingfactor)
    learner.α_eff = learner.α * modulatingfactor
end
