%% Data Simulation
function data = simulations_2(theta, sim_size) % feed the learning rate & inverse temp.

    switching = 1;
    eps = theta(1); % learning rate
    rho = theta(2); % inverse temperature
    V_A_2 = zeros(sim_size,1);
    V_B_2 = zeros(sim_size,1);
    data.choices = []; % initiate empty choices array
    data.rewards = []; % initiate empty reward array
    
    for t = 1:sim_size % number of trials
        
        % patients choice simulation
        % start
        p_choose_B = exp(V_B_2(t)) / (exp(V_B_2(t)) + exp(V_A_2(t))); 
        c = 1 + (p_choose_B > rand); % choice
        % end
        
        % Reward mechanism 
        % start
        if switching  == 1
            if c==1
               r = (0.45 > rand);
               V_A_2(t+1) = V_A_2(t) + eps * ((rho*r) - V_A_2(t)); % value update
               V_B_2(t+1) = V_B_2(t);
            end
            if c==2
               r = (0.70 > rand);
               V_B_2(t+1) = V_B_2(t) + eps * ((rho*r) - V_B_2(t)); % value update
               V_A_2(t+1) = V_A_2(t);
            end
            if mod(t,25) == 0
                switching = 0;
            end
        else 
            if c==1
               r = (0.70 > rand);
               V_A_2(t+1) = V_A_2(t) + eps * ((rho*r) - V_A_2(t)); % value update
               V_B_2(t+1) = V_B_2(t);
            end
            
            if c==2
               r = (0.30 > rand);
               V_B_2(t+1) = V_B_2(t) + eps * ((rho*r) - V_B_2(t)); % value update
               V_A_2(t+1) = V_A_2(t);
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
    data.values_A = V_A_2(1:sim_size)';
    data.values_B = V_B_2(1:sim_size)';
    data.rewarded = data.rewards;
    data.reward_count = size(data.rewarded(data.rewarded == 1), 2);
end