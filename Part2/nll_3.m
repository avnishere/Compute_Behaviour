%% Negative Log Likelihood

function nll_sum3 = nll_3(data, theta)
    eps = logsig(theta(1));
    beta = theta(2);
    V = [0.5, 0.5];
    num_trials = numel(data.choices);
    choice_probabilities = nan(num_trials, 1);
    for t = 1:num_trials
        c = data.choices(t); 
        c_alt = 1 + (c==1);
        p_choose_c = exp(beta*V(c)) / (exp(beta*V(c)) + exp(beta*V(c_alt)));
        choice_probabilities(t) = p_choose_c;
        V(c) = V(c) + eps * (data.rewards(t) - V(c));
    end
    nll_sum3 = -sum(log(choice_probabilities));
end