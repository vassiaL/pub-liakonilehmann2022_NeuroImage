using Distributions, SpecialFunctions, Random
"""
Estimation of the transition matrix for outlier task via a Particle filter
"""
struct TParticleFilterJump <: TPs1a0s0
    nr_states::Int
    nr_actions::Int
    nparticles::Int # Per state and action
    Neffthrs::Float64 # Neff = approx nr of particles that have a weight which
                    # meaningfully contributes to the probability distribution.
    surpriseprobability::Float64
    stochasticity::Float64
    particlesjump::Array{Bool, 3} # wannabe nr_states x nr_actions x nparticles.
    weights::Array{Float64, 3} # wannabe nr_states x nr_actions x nparticles.
    Sgmparticles::Array{Float64, 1} # wannabe nparticles.
    Sgm::Array{Float64, 1}
    m::Float64
    γ_surprise::Array{Float64, 1}
    Ps1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    SPE::Array{Float64, 1}
    counts::Array{Array{Float64,1}, 3}
    seed::Any
    rng::MersenneTwister
    terminalstates::Array{Int,1}
end
function TParticleFilterJump(;nr_states = 10, nr_actions = 4, nparticles = 6,
                        surpriseprobability = .01,
                        stochasticity = .01, seed = 3)
    Neffthrs = nparticles/2.
    particlesjump = Array{Bool, 3}(undef, nr_actions, nr_states, nparticles)
    weights = Array{Float64, 3}(undef, nr_actions, nr_states, nparticles)
    Sgmparticles = Array{Float64, 1}(undef, nparticles)
    Ps1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:nr_states]
    [Ps1a0s0[sprime][(a, s)] = 1. /nr_states for sprime in 1:nr_states for a in 1:nr_actions for s in 1:nr_states]
    counts = Array{Array{Float64,1}}(undef, nr_actions, nr_states, nparticles)
    rng = MersenneTwister(seed)
    [weights[a0, s0, i] = 1. /nparticles for a0 in 1:nr_actions for s0 in 1:nr_states for i in 1:nparticles]
    Sgmparticles .= 1.; Sgm = [1.]
    [counts[a0, s0, i] = zeros(nr_states) for a0 in 1:nr_actions for s0 in 1:nr_states for i in 1:nparticles]
    m = surpriseprobability/ (1. - surpriseprobability)
    γ_surprise = [m/(1. + m)]
    SPE = [1. - 1. / nr_states]

    TParticleFilterJump(nr_states, nr_actions, nparticles, Neffthrs, surpriseprobability, stochasticity,
                    particlesjump, weights, Sgmparticles, Sgm, m, γ_surprise, Ps1a0s0, SPE, counts, seed, rng, Int[])
end
export TParticleFilterJump
function updatet!(learnerT::TParticleFilterJump, s0, a0, s1, done)
    learnerT.particlesjump[a0, s0, :] .= false
    calcSPE!(learnerT, s0, a0, s1) # Calculate a hypothetical SPE
    calcSgm!(learnerT, s0, a0, s1)
    calcγ!(learnerT)
    stayterms = computestayterms(learnerT, s0, a0, s1)
    getweights!(learnerT, s0, a0, stayterms, 1. / learnerT.nr_states)
    sampleparticles!(learnerT, s0, a0, stayterms, 1. / learnerT.nr_states)
    Neff = 1. /sum(learnerT.weights[a0, s0, :] .^2)
    if Neff <= learnerT.Neffthrs; resample!(learnerT, s0, a0); end
    updatecounts!(learnerT, s0, a0, s1)
    computePs1a0s0!(learnerT, s0, a0)
    computeterminalPs1a0s0!(learnerT, s1, done)
end
export updatet!
function computestayterms(learnerT::TParticleFilterJump, s0, a0, s1)
    stayterms = zeros(learnerT.nparticles)
    for i in 1:learnerT.nparticles
        stayterms[i] = (learnerT.stochasticity + learnerT.counts[a0, s0, i][s1])
        stayterms[i] /= sum(learnerT.stochasticity .+ learnerT.counts[a0, s0, i])
    end
    stayterms
end
export computestayterms
function getweights!(learnerT::TParticleFilterJump, s0, a0, stayterms, jumpterm)
    for i in 1:learnerT.nparticles #firstratio = B(s + a(h_t-1)') / B(s + a(h_t-1)). secondratio = B(s + a(h_t=h_t-1 + 1)) / B(s)
        particleweightupdate = (1. - learnerT.surpriseprobability) * stayterms[i]
        particleweightupdate += learnerT.surpriseprobability * jumpterm
        learnerT.weights[a0, s0, i] *= particleweightupdate
    end

    learnerT.weights[a0, s0, :] ./= sum(learnerT.weights[a0, s0, :]) # Normalize

end
export getweights!
function sampleparticles!(learnerT::TParticleFilterJump,
                        s0, a0, stayterms, jumpterm)
    for i in 1:learnerT.nparticles
        particlestayprobability = computeproposaldistribution(learnerT, stayterms[i], jumpterm)

        r = rand(learnerT.rng) # Draw and possibly update
        if r > particlestayprobability
            # println("jump!")
            learnerT.particlesjump[a0, s0, i] = true
        end
    end
end
export sampleparticles!
function computeproposaldistribution(learnerT::TParticleFilterJump,
                                    istayterm, jumpterm)
    particlestayprobability = 1. /(1. + ((learnerT.surpriseprobability * jumpterm) /
                                ((1. - learnerT.surpriseprobability) * istayterm)))
end
export computeproposaldistribution
function updatecounts!(learnerT::TParticleFilterJump, s0, a0, s1)
    for i in 1:learnerT.nparticles
        if !learnerT.particlesjump[a0, s0, i] # if it is not a surprise trial: Integrate
            learnerT.counts[a0, s0, i][s1] += 1 # +1 for s'
        end
    end
end
export updatecounts!
function resample!(learnerT::TParticleFilterJump, s0, a0)
    tempcopyparticleweights = deepcopy(learnerT.weights[a0, s0, :])
    d = Categorical(tempcopyparticleweights)
    tempcopyparticlesjump = copy(learnerT.particlesjump[a0, s0, :])
    tempcopycounts = deepcopy(learnerT.counts[a0, s0, :])
    for i in 1:learnerT.nparticles
        sampledindex = rand(learnerT.rng, d)
        learnerT.particlesjump[a0, s0, i] = copy(tempcopyparticlesjump[sampledindex])
        learnerT.weights[a0, s0, i] = 1. /learnerT.nparticles
        learnerT.counts[a0, s0, i] = deepcopy(tempcopycounts[sampledindex])
    end
end
export resample!
function computePs1a0s0!(learnerT::TParticleFilterJump, s0, a0)
    thetasweighted = zeros(learnerT.nparticles, learnerT.nr_states)
    for i in 1:learnerT.nparticles
        thetas = (learnerT.stochasticity .+ learnerT.counts[a0, s0, i])
        thetas /= sum(learnerT.stochasticity .+ learnerT.counts[a0, s0, i])
        thetasweighted[i,:] = learnerT.weights[a0, s0, i] .* thetas
    end
    expectedvaluethetas = sum(thetasweighted, dims = 1)
    for s in 1:learnerT.nr_states
        learnerT.Ps1a0s0[s][(a0, s0)] = copy(expectedvaluethetas[s])
        # @show learnerT.Ps1a0s0[s][(a0, s0)]
    end
end
""" Outgoing transitions from goal state: set all to 0, except for 1 to itself
(absorbing state) """
function computeterminalPs1a0s0!(learnerT::TPs1a0s0, s1, done)
    if done
        if !in(s1, learnerT.terminalstates)
            push!(learnerT.terminalstates, s1)
            for a in 1:learnerT.nr_actions
                for s in 1:learnerT.nr_states
                    if s == s1
                        learnerT.Ps1a0s0[s][(a, s1)] = 1.
                    else
                        learnerT.Ps1a0s0[s][(a, s1)] = 0.
                    end
                end
            end
        end
    end
end
function calcSgmi(αt, α0, s1)
    p0 = α0[s1] / sum(α0)
    pt = αt[s1] / sum(αt)
    p0 / pt
end
function calcSgmparticles!(learnerT::TParticleFilterJump, s0, a0, s1)
    for i in 1:learnerT.nparticles
        learnerT.Sgmparticles[i] = calcSgmi(
                            learnerT.stochasticity .+ learnerT.counts[a0, s0, i],
                            learnerT.stochasticity .* ones(learnerT.nr_states),
                            s1)
    end
end
function calcSgm!(learnerT::TParticleFilterJump, s0, a0, s1)

    calcSgmparticles!(learnerT, s0, a0, s1)

    if any(isequal.(learnerT.Sgmparticles, 0.))
        learnerT.Sgm[1] = 0.
    else
        SgmInvWeighted = learnerT.weights[a0, s0, :] ./ learnerT.Sgmparticles
        learnerT.Sgm[1] = 1. / sum(SgmInvWeighted)
    end
end
function calcγ!(learnerT::TParticleFilterJump)
    learnerT.γ_surprise[1] = learnerT.m * learnerT.Sgm[1]/(1. + learnerT.m * learnerT.Sgm[1])
end
function calcExpectedHiddenState(learnerT::TParticleFilterJump, s0, a0)
    expectedhiddenstate = sum(Float64.(learnerT.particlesjump[a0, s0, :]) .*
                        learnerT.weights[a0, s0, :])
    expectedhiddenstate = round(expectedhiddenstate)
end
function calcExpectedHiddenStateNotRounded(learnerT::TParticleFilterJump, s0, a0)
    expectedhiddenstate = sum(Float64.(learnerT.particlesjump[a0, s0, :]) .*
                        learnerT.weights[a0, s0, :])
end
function calcSPE!(learnerT::TParticleFilterJump, s0, a0, s1)
    learnerT.SPE[1] = 1. - learnerT.Ps1a0s0[s1][(a0, s0)]
end
