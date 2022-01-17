"""
RIntegratorNextStateReward: Estimate Rsas'
REstimateDummy: Do nothing - used for original SmallBackups implementation of Prioritized Sweeping
RIntegratorStateActionReward: Estimate Rsa (when reward is a function of current s-a pair and NOT s')
"""
struct RIntegratorNextStateReward
    nr_states::Int
    init_R::Float64
    Ns::Array{Int, 1}
    Rsum::Array{Float64, 1}
    R::Array{Float64, 1}
end
function RIntegratorNextStateReward(; nr_states = 10, init_R = 0.)
    Ns = zeros(Int, nr_states)
    Rsum = zeros(nr_states)
    R = zeros(nr_states) .+ init_R
    RIntegratorNextStateReward(nr_states, init_R, Ns, Rsum, R)
end
export RIntegratorNextStateReward
function updater!(learnerR::RIntegratorNextStateReward, s0, a0, s1, r)
    learnerR.Ns[s1] += 1
    learnerR.Rsum[s1] += r
    learnerR.R[s1] = learnerR.Rsum[s1] / learnerR.Ns[s1]
end

struct REstimateDummy end
REstimateDummy()
updater!(::REstimateDummy, s0, a0, s1, r) = nothing
export REstimateDummy

struct RIntegratorStateActionReward
    nr_states::Int
    nr_actions::Int
    init_R::Float64
    Nsa::Array{Int, 2}
    Rsum::Array{Float64, 2}
    R::Array{Float64, 2}
end
function RIntegratorStateActionReward(; nr_states = 10, nr_actions = 4, init_R = 0.)
    Nsa = zeros(Int, nr_actions, nr_states)
    Rsum = zeros(nr_actions, nr_states)
    R = zeros(nr_actions, nr_states) .+ init_R
    RIntegratorStateActionReward(nr_states, nr_actions, init_R, Nsa, Rsum, R)
end
export RIntegratorStateActionReward
function updater!(learnerR::RIntegratorStateActionReward, s0, a0, s1, r)
    learnerR.Nsa[a0, s0] += 1
    learnerR.Rsum[a0, s0] += r
    learnerR.R[a0, s0] = learnerR.Rsum[a0, s0] / learnerR.Nsa[a0, s0]
    # learnerR.R[a0, s0] -= learnerR.R[a0, s0] / learnerR.Nsa[a0, s0]
    # learnerR.R[a0, s0] += r / learnerR.Nsa[a0, s0]
end

export updater!
