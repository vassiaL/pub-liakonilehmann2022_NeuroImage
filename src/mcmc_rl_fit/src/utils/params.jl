# Parameter sampling distributions

using Distributions

struct MetropolisParam
    name::AbstractString
    descr::AbstractString
    prior_Distribution::Distribution
    rwalk_step_size_std::Float64
end

# a helper to get priors for sampling posterior distributions.
# the idea is to keep the original definition (identified by param_idx) while
# setting all others to a narrow gaussian around the MAP_values
# NOT USED eventually!
function get_prior_for_cond_sampling(orig_prior:: Array, MAP_values::Array, param_idx::Int; std_posterior=0.01, rwalk_step_size_std = 0.0001)
p_post = deepcopy(orig_prior)
    for (i, p) in enumerate(orig_prior)
        if i == param_idx
            # keep the original definition
        else
            # create a new, narrow gaussian, centered at MAP_values[i]
            # Truncate to the same interval as the orig distribution
            param_def = MetropolisParam(
                p.name,
                p.descr,
                Distributions.TruncatedNormal(MAP_values[i], std_posterior, minimum(p.prior_Distribution), maximum(p.prior_Distribution)) ,
                rwalk_step_size_std
                )
            # print(param_def)
            p_post[i] = param_def
        end
    end
p_post
end

function param_Uniform(name, lo=0.0, hi=1. ; description="generic param", rwalk_step_size_std = 0.008)
    prior = Distributions.Uniform(lo, hi)
    MetropolisParam(name, description, prior, rwalk_step_size_std)
end

# some helpers to quickly get a reasonable default
function paramAlpha(lo=0.0, hi=1.; name="alpha", rwalk_step_size_std = 0.008)
    paramAlpha(Distributions.Uniform(lo, hi); name=name,rwalk_step_size_std = rwalk_step_size_std)
end
function paramAlpha(prior::Distribution ;name="alpha", rwalk_step_size_std = 0.008)
    MetropolisParam(name, "learning rate", prior, rwalk_step_size_std)
end

function paramEta(lo=0.0, hi=1.; name="eta", rwalk_step_size_std = 0.008)
    paramEta(Distributions.Uniform(lo, hi); name=name, rwalk_step_size_std = rwalk_step_size_std)
end
function paramEta(prior::Distribution; name="eta", rwalk_step_size_std = 0.008)
    MetropolisParam(name, "learning rate", prior, rwalk_step_size_std)
end

function paramPhi(lo=0.0, hi=1.; rwalk_step_size_std = 0.008)
    paramPhi(Distributions.Uniform(lo, hi); rwalk_step_size_std = rwalk_step_size_std)
end
function paramPhi(prior::Distribution;rwalk_step_size_std = 0.008)
    MetropolisParam("phi", "surprise trial learning rate", prior, rwalk_step_size_std)
end

function paramExpDecayOffset(lo=0, hi=1.; rwalk_step_size_std = 0.008)
    paramExpDecayOffset(Distributions.Uniform(lo, hi); rwalk_step_size_std = rwalk_step_size_std)
end
function paramExpDecayOffset(prior::Distribution; rwalk_step_size_std = 0.008)
    MetropolisParam("decay_offset", "exp decay offset", prior, rwalk_step_size_std)
end

function paramExpDecaySlope(lo= -0.10, hi=+1.0; rwalk_step_size_std = 0.004)
    #note the -.1 exponent would allow for INCREASING m-b contribution. exp(-t*slope)
    paramExpDecaySlope(Distributions.Uniform(lo, hi); rwalk_step_size_std = rwalk_step_size_std)
end
function paramExpDecaySlope(prior::Distribution; rwalk_step_size_std = 0.004)
    MetropolisParam("decay_slope", "exp decay slope", prior, rwalk_step_size_std)
end


function paramGamma(lo=0.0, hi=1.; rwalk_step_size_std = 0.008)
    paramGamma(Distributions.Uniform(lo, hi); rwalk_step_size_std = rwalk_step_size_std)
end
function paramGamma(prior::Distribution; rwalk_step_size_std = 0.008)
    MetropolisParam("gamma", "discount factor", prior, rwalk_step_size_std)
end
function paramLambda(lo=0., hi=1.; name="lambda", rwalk_step_size_std = 0.008)
    paramLambda(Distributions.Uniform(lo, hi); name=name, rwalk_step_size_std = rwalk_step_size_std)
end

function paramLambda(prior::Distribution; name="lambda", rwalk_step_size_std = 0.008)
    MetropolisParam(name, "eligibility trace", prior, rwalk_step_size_std)
end

function paramETTau(lo=0., hi=40.; name="ET_tau", rwalk_step_size_std = 0.05)
    paramETTau(Distributions.Uniform(lo, hi); name=name, rwalk_step_size_std = rwalk_step_size_std)
end
function paramETTau(prior::Distribution; name="ET_tau", rwalk_step_size_std = 0.05)
    MetropolisParam(name, "eligibility trace", prior, rwalk_step_size_std)
end

function paramTemp(lo=0.01, hi=1.25; rwalk_step_size_std = 0.05) #rwalk_step_size_std = 0.008)
    paramTemp(Distributions.Uniform(lo, hi); rwalk_step_size_std = rwalk_step_size_std)
end
function paramTemp(prior::Distribution; rwalk_step_size_std = 0.05) #rwalk_step_size_std = 0.008)
    MetropolisParam("temp", "softmax temperature", prior, rwalk_step_size_std)
end

function paramBias(lo=0.0, hi=0.6; rwalk_step_size_std = 0.004)
    paramBias(Distributions.Uniform(lo, hi); rwalk_step_size_std = rwalk_step_size_std)
end

function paramBias(prior::Distribution; rwalk_step_size_std = 0.004)
    MetropolisParam("bias", "left/right bias", prior, rwalk_step_size_std)
end

function paramAnnealingSlope(lo=0.0, hi=0.2; name = "annealing", rwalk_step_size_std = 0.004)
    paramAnnealingSlope(Distributions.Uniform(lo, hi); name=name, rwalk_step_size_std = rwalk_step_size_std)
end

function paramAnnealingSlope(prior::Distribution; name = "annealing", rwalk_step_size_std = 0.004)
    MetropolisParam(name, "slope s in alpha*exp(-t*s)", prior, rwalk_step_size_std)
end

function paramSurpriseProbability(lo=0., hi=1.; name="surpriseprobability", rwalk_step_size_std = 0.004)
    paramSurpriseProbability(Distributions.Uniform(lo, hi); name=name, rwalk_step_size_std = rwalk_step_size_std)
end
function paramSurpriseProbability(prior::Distribution; name="surpriseprobability", rwalk_step_size_std = 0.004)
    MetropolisParam(name, "surpriseprobability in particle filter", prior, rwalk_step_size_std)
end

function paramStochasticity(lo=0., hi=20.; name="stochasticity", rwalk_step_size_std = 0.05)
    paramStochasticity(Distributions.Uniform(lo, hi); name=name, rwalk_step_size_std = rwalk_step_size_std)
end
function paramStochasticity(prior::Distribution; name="stochasticity", rwalk_step_size_std = 0.05)
    MetropolisParam(name, "stochasticity in particle filter", prior, rwalk_step_size_std)
end

function paramEtaleak(lo=0., hi=1.; name="etaleak", rwalk_step_size_std = 0.008)
    paramEtaleak(Distributions.Uniform(lo, hi); name=name, rwalk_step_size_std = rwalk_step_size_std)
end
function paramEtaleak(prior::Distribution; name="etaleak", rwalk_step_size_std = 0.008)
    MetropolisParam(name, "etaleak in leaky Tsas' estimator", prior, rwalk_step_size_std)
end


function paramNrUpdateCycles(lo=0, hi=9; name="nr_upd_cycles", rwalk_step_size_std = 1.)
    paramNrUpdateCycles(Distributions.DiscreteUniform(lo, hi); name=name, rwalk_step_size_std = rwalk_step_size_std)
end
function paramNrUpdateCycles(prior::Distribution; name="nr_upd_cycles", rwalk_step_size_std = 1.)
    MetropolisParam(name, "Nr of update cycles in Prioritized Sweeping", prior, rwalk_step_size_std)
end


function paramNParticles(lo=1, hi=15; name="nparticles", rwalk_step_size_std = 1.)
    paramNParticles(Distributions.DiscreteUniform(lo, hi); name=name, rwalk_step_size_std = rwalk_step_size_std)
end
function paramNParticles(prior::Distribution; name="nparticles", rwalk_step_size_std = 1.)
    MetropolisParam(name, "Nr of particles in Particle Filter transition learner", prior, rwalk_step_size_std)
end

function paramAlpha_0(lo=0.0, hi=1.; name="alpha_0", rwalk_step_size_std = 0.008)
    paramAlpha_0(Distributions.Uniform(lo, hi); name=name,rwalk_step_size_std = rwalk_step_size_std)
end
function paramAlpha_0(prior::Distribution ;name="alpha_0", rwalk_step_size_std = 0.008)
    MetropolisParam(name, "learning rate constant term", prior, rwalk_step_size_std)
end

function paramAlpha_surprise(lo=0.0, hi=1.; name="alpha_surprise", rwalk_step_size_std = 0.008)
    paramAlpha_surprise(Distributions.Uniform(lo, hi); name=name,rwalk_step_size_std = rwalk_step_size_std)
end
function paramAlpha_surprise(prior::Distribution ;name="alpha_surprise", rwalk_step_size_std = 0.008)
    MetropolisParam(name, "learning rate modulating term", prior, rwalk_step_size_std)
end

function paramWeight(lo=0.0, hi=20.; name="w_MB", rwalk_step_size_std = 0.05)
    paramAlpha_surprise(Distributions.Uniform(lo, hi); name=name,rwalk_step_size_std = rwalk_step_size_std)
end
function paramWeight(prior::Distribution ;name="w_MB", rwalk_step_size_std = 0.05)
    MetropolisParam(name, "weight in hybrid", prior, rwalk_step_size_std)
end

function paramActionCost(lo=0.0, hi=5.; name="actioncost", rwalk_step_size_std = 0.05)
    paramActionCost(Distributions.Uniform(lo, hi); name=name, rwalk_step_size_std = rwalk_step_size_std)
end
function paramActionCost(prior::Distribution ;name="actioncost", rwalk_step_size_std = 0.05)
    MetropolisParam(name, "action cost", prior, rwalk_step_size_std)
end
