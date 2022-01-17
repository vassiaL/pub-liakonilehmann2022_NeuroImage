function learnerT = TParticleFilterJump_update(learnerT, s0, a0, s1, done)
    learnerT.particlesjump(a0, s0, :) = 0;
    learnerT = calcSPE(learnerT, s0, a0, s1); % Calculate a hypothetical SPE
    learnerT = calcSgm(learnerT, s0, a0, s1);
    learnerT = calcgamma(learnerT);
    stayterms = computestayterms(learnerT, s0, a0, s1);
    learnerT = getweights(learnerT, s0, a0, stayterms, 1. / learnerT.nr_states);
    learnerT = sampleparticles(learnerT, s0, a0, stayterms, 1. / learnerT.nr_states);
    Neff = 1. /sum(learnerT.weights(a0, s0, :) .^2);
    if Neff <= learnerT.Neffthrs 
        learnerT = resample(learnerT, s0, a0); 
    end
    
    learnerT = updatecounts(learnerT, s0, a0, s1);
    learnerT = computePs1a0s0(learnerT, s0, a0);
    learnerT = computeterminalPs1a0s0(learnerT, s1, done);
end

function stayterms = computestayterms(learnerT, s0, a0, s1)
    stayterms = zeros(1, learnerT.nparticles);
    for i=1:learnerT.nparticles
        stayterms(i) = (learnerT.stochasticity + learnerT.counts(a0, s0, i, s1));
        stayterms(i) = stayterms(i) / sum(learnerT.stochasticity + learnerT.counts(a0, s0, i, :));
    end
    
end

function learnerT = getweights(learnerT, s0, a0, stayterms, jumpterm)
    for i=1:learnerT.nparticles % firstratio = B(s + a(h_t-1)') / B(s + a(h_t-1)). secondratio = B(s + a(h_t=h_t-1 + 1)) / B(s)
        particleweightupdate = (1. - learnerT.surpriseprobability) * stayterms(i);
        particleweightupdate = particleweightupdate + learnerT.surpriseprobability * jumpterm;
        learnerT.weights(a0, s0, i) = learnerT.weights(a0, s0, i) * particleweightupdate;
    end
    learnerT.weights(a0, s0, :) = learnerT.weights(a0, s0, :) / sum(learnerT.weights(a0, s0, :)); % Normalize
end

function learnerT = sampleparticles(learnerT, s0, a0, stayterms, jumpterm)
    for i=1:learnerT.nparticles
        particlestayprobability = computeproposaldistribution(learnerT, stayterms(i), jumpterm);
        r = rand(learnerT.rgen); % Draw and possibly update
        if r > particlestayprobability
            learnerT.particlesjump(a0, s0, i) = 1;
        end
    end
end

function particlestayprobability = computeproposaldistribution(learnerT, istayterm, jumpterm)
    particlestayprobability = 1. /(1. + ((learnerT.surpriseprobability * jumpterm) / ...
                                ((1. - learnerT.surpriseprobability) * istayterm)));
end

function learnerT = updatecounts(learnerT, s0, a0, s1)
    for i = 1:learnerT.nparticles
        if ~learnerT.particlesjump(a0, s0, i) % if it is not a surprise trial: Integrate
            learnerT.counts(a0, s0, i, s1) = learnerT.counts(a0, s0, i, s1) + 1; % +1 for s'
        end
    end
end

function learnerT = resample(learnerT, s0, a0)
    tempcopyparticleweights = learnerT.weights(a0, s0, :);
%     d = Categorical(tempcopyparticleweights);
% datasample(s,1:3,1, 'Weights',[0. 0.1 0.9])
     tempcopyparticlesjump = squeeze(learnerT.particlesjump(a0, s0, :));
     tempcopycounts = squeeze(learnerT.counts(a0, s0, :, :));
    for i=1:learnerT.nparticles
        sampledindex = datasample(learnerT.rgen,1:learnerT.nparticles, 1, 'Weights', squeeze(tempcopyparticleweights)'); %rand(learnerT.rng, d)
        learnerT.particlesjump(a0, s0, i) = tempcopyparticlesjump(sampledindex);
        learnerT.weights(a0, s0, i) = 1. /learnerT.nparticles;
        learnerT.counts(a0, s0, i, :) = tempcopycounts(sampledindex, :);
    end
end

function learnerT = computePs1a0s0(learnerT, s0, a0)
    thetasweighted = zeros(learnerT.nparticles, learnerT.nr_states);
    for i=1:learnerT.nparticles
        % thetas = (learnerT.stochasticity .+ learnerT.counts[a0, s0, i])/sum(learnerT.stochasticity .+ learnerT.counts[a0, s0, i])
        thetas = (learnerT.stochasticity + learnerT.counts(a0, s0, i, :)) ...
                / sum(learnerT.stochasticity + learnerT.counts(a0, s0, i, :));
        thetasweighted(i,:) = learnerT.weights(a0, s0, i) .* thetas;
    end
    expectedvaluethetas = sum(thetasweighted, 1);
    for s=1:learnerT.nr_states
        learnerT.Ps1a0s0(a0, s0, s) = expectedvaluethetas(s);
    end
end
% """ Outgoing transitions from goal state: set all to 0, except for 1 to itself
% (absorbing state) """
function learnerT = computeterminalPs1a0s0(learnerT, s1, done)
    if done
        if ~ismember(s1, learnerT.terminalstates)
            learnerT.terminalstates(s1) = s1;
            for a=1:learnerT.nr_actions
                for s=1:learnerT.nr_states
                    if s == s1
                        learnerT.Ps1a0s0(a, s1, s) = 1.;
                    else
                        learnerT.Ps1a0s0(a, s1, s) = 0.;
                    end
                end
            end
        end
    end
end

function Sgmi = calcSgmi(alphat, alpha0, s1)
    p0 = alpha0(s1) / sum(alpha0);
    pt = alphat(s1) / sum(alphat);
    Sgmi = p0 / pt;
end

function learnerT = calcSgmparticles(learnerT, s0, a0, s1)
    for i=1:learnerT.nparticles
        learnerT.Sgmparticles(i) = calcSgmi(...
                            squeeze(learnerT.stochasticity + learnerT.counts(a0, s0, i, :)), ...
                            learnerT.stochasticity * ones(1, learnerT.nr_states), ...
                            s1);
%         #learnerT.Sgmparticles[i] = checkInf(learnerT.Sgmparticles[i])
    end
end
function learnerT = calcSgm(learnerT, s0, a0, s1)

    learnerT = calcSgmparticles(learnerT, s0, a0, s1);

    if any(learnerT.Sgmparticles == 0.)
        learnerT.Sgm = 0.;
    else
        SgmInvWeighted = squeeze(learnerT.weights(a0, s0, :)) ./ learnerT.Sgmparticles';
        learnerT.Sgm = 1. / sum(SgmInvWeighted);
    end
end
function learnerT = calcgamma(learnerT)
    learnerT.gamma_surprise = learnerT.m * learnerT.Sgm/(1. + learnerT.m * learnerT.Sgm);
end
% function expectedhiddenstate = calcExpectedHiddenState(learnerT, s0, a0)
%     expectedhiddenstate = sum(learnerT.particlesjump(a0, s0, :) .* learnerT.weights(a0, s0, :));
% %     # expectedhiddenstate = Bool(round(expectedhiddenstate))
% %     # @show expectedhiddenstate
%     expectedhiddenstate = round(expectedhiddenstate);
% end
% function expectedhiddenstate = calcExpectedHiddenStateNotRounded(learnerT, s0, a0)
%     expectedhiddenstate = sum(learnerT.particlesjump(a0, s0, :) .* learnerT.weights(a0, s0, :));
% %     # expectedhiddenstate = Bool(round(expectedhiddenstate))
% end
function learnerT = calcSPE(learnerT, s0, a0, s1)
    learnerT.SPE = 1. - learnerT.Ps1a0s0(a0, s0, s1);
end
