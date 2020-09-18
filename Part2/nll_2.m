%% Negative Log Likelihood

function nll_sum2 = nll_2(data, theta)
    eps = logsig(theta(1));
    rho = theta(2);
    V = [0, 0];
    num_trials = numel(data.choices);
    choice_probabilities = nan(num_trials, 1);
    for t = 1:num_trials
        c = data.choices(t); 
        c_alt = 1 + (c==1);
        p_choose_c = exp(V(c)) / (exp(V(c)) + exp(V(c_alt)));
        choice_probabilities(t) = p_choose_c;
        V(c) = V(c) + eps *((rho*data.rewards(t)) - V(c));
    end
    nll_sum2 = -sum(log(choice_probabilities));
end
