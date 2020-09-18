function [f, g] = nll_withgradient(data, theta)

    eps = logsig(theta(1));
    beta = theta(2);
    V = [0, 0];
    num_trials = numel(data.choices);
    choice_probabilities = nan(num_trials, 1);
    grad = nan(num_trials, 1);
    for t = 1:num_trials
        c = data.choices(t); 
        c_alt = 1 + (c==1);
        p_choose_c = exp(beta*V(c)) / (exp(beta*V(c)) + exp(beta*V(c_alt)));
        choice_probabilities(t) = p_choose_c;
        V(c) = V(c) + eps * (data.rewards(t) - V(c));
        
        grad(t) = (2*V(c)*exp(beta*V(c))) + V(c)*exp(beta*V(c_alt)) + V(c_alt)*exp(beta*V(c_alt)); 
    end
    f = -sum(log(choice_probabilities));
    if nargout > 1 % gradient required
    g = [-sum(grad); 0];
    end
    
end