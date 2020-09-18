
%% Data Simulation
function data = simulations(theta, sim_size) % feed the learning rate & inverse temp.

    switching = 1;
    eps = theta(1); % learning rate
    beta = theta(2); % inverse temperature
    V_A = zeros(sim_size,1);
    V_B = zeros(sim_size,1);
    data.choices = []; % initiate empty choices array
    data.rewards = []; % initiate empty reward array
    
    for t = 1:sim_size % number of trials
        
        % patients choice simulation
        % start
        p_choose_B = exp(beta*V_B(t)) / (exp(beta*V_B(t)) + exp(beta*V_A(t))); 
        c = 1 + (p_choose_B > rand); % choice
        % end
        
        % Reward mechanism 
        % start
        if switching  == 1
            if c==1
               r = (0.45 > rand);
               V_A(t+1) = V_A(t) + eps * (r - V_A(t)); % value update
               V_B(t+1) = V_B(t);
            end
            if c==2
               r = (0.70 > rand);
               V_B(t+1) = V_B(t) + eps * (r - V_B(t)); % value update
               V_A(t+1) = V_A(t);
            end
            if mod(t,25) == 0
                switching = 0;
            end
        else 
            if c==1
               r = (0.70 > rand);
               V_A(t+1) = V_A(t) + eps * (r - V_A(t)); % value update
               V_B(t+1) = V_B(t);
            end
            
            if c==2
               r = (0.30 > rand);
               V_B(t+1) = V_B(t) + eps * (r - V_B(t)); % value update
               V_A(t+1) = V_A(t);
            end
            if mod(t,25) == 0
                switching = 1;
            end
            
        end
        % end
        
         % storing choices, rewards 
        data.choices(t) = c;
        data.rewards(t) = r;
    end
    % storing values 
    data.values_A = V_A(1:sim_size)';
    data.values_B = V_B(1:sim_size)';
    data.rewarded = data.rewards;
    data.VA_VB = data.values_A - data.values_B;
    data.reward_count = size(data.rewarded(data.rewarded == 1), 2);
end