
%% Main File
clear all;
close all;
clc;

%% Task(a)

% simulating 250 choices with [eps beta] as [0.35 5.5]
eps = 0.35;
beta = 5.5;
theta_task_a = [eps beta];
sim_trials = 250;
task_a = simulations(theta_task_a, sim_trials); % the function

% start plotting figures
figure(1)
plot(task_a.values_A);
title('Evolution of Values for stimulus A');
xlabel('Trial Number');
ylabel('Value associted with A or V(A)');

figure(2)
plot(task_a.values_B);
title('Evolution of Values for stimulus B');
xlabel('Trial Number');
ylabel('Value associted with B or V(B)');

figure(3)
val_diff = (task_a.values_A - task_a.values_B);
plot(val_diff);
title('Evolution of Difference of Values for A nd B');
xlabel('Trial Number');
ylabel('Difference Value or V(A) - V(B)');
% end plotting figures

% multiple simulations to get the average reward
task1_runs = 1000;

val_A = zeros(task1_runs, sim_trials);
val_B = zeros(task1_runs, sim_trials);

% start iterations for multiple simulations
for i = 1:task1_runs
    
generate_data_task_a = simulations(theta_task_a, sim_trials);
val_A(i,:) = generate_data_task_a.values_A;
val_B(i,:) = generate_data_task_a.values_B;
reward_counts(i) = generate_data_task_a.reward_count;

end

% end iterations for multiple simulations

mean_reward_task_a = mean(reward_counts);
disp('The average reward is:');
disp(mean_reward_task_a);

% start checking mean reward for random response

val_A_rand = zeros(task1_runs, sim_trials);
val_B_rand = zeros(task1_runs, sim_trials);

for i = 1:task1_runs
    
generate_data_rand = simulations_rand(theta_task_a, sim_trials);
val_A_rand(i,:) = generate_data_rand.values_A;
val_B_rand(i,:) = generate_data_rand.values_B;
reward_counts_rand(i) = generate_data_rand.reward_count;

end

mean_reward_task_a_rand = mean(reward_counts_rand);
disp('The average reward for random choice is:');
disp(mean_reward_task_a_rand);

% end checking mean reward for random response

%% Task(b)

% start running simulation over combinations of eps and beta values

eps_range = -1:0.1:1;
beta_range = -10:0.25:15;

for i=1: size(eps_range,2)
    theta_task_b(1) = eps_range(i);
    
    for j = 1: size(beta_range,2)
        theta_task_b(2)  = beta_range(j);
        
        for k = 1:1000
            generate_data_task_b = simulations(theta_task_b, sim_trials);
            reward_counts(k) = generate_data_task_b.reward_count;
        end
        mean_reward_task_b(i,j) = mean(reward_counts);

    end
        
end

figure(4)
surf(beta_range, eps_range, mean_reward_task_b);
title('Average Rewards as a function of Parameters');
ylabel('The learning Rate');
xlabel('The Inverse Temperature');
zlabel('Average Reward Count');

% end running simulation over combinations of eps and beta values

%% Task (c) Likelihood Function

%loading the participants data
load('data.mat');

%Hint Verification
data_part_1.choices = data.choices(1,:);
data_part_1.rewards = data.rewards(1,:);
nll_participant_1 = nll(data_part_1, [0.4 5]);
disp('The verification NLL value for Participant 1 is:');
disp(nll_participant_1);

data_part_2.choices = data.choices(2,:);
data_part_2.rewards = data.rewards(2,:);
nll_participant_2= nll(data_part_2, [0.4 5]);

disp('The NLL for Participant 2 is:');
disp(nll_participant_2);

% Task(d) Model Fitting

theta_zero = [0 0]; % initializing with zero valued parameters

for ind = 1: size(data.choices, 1)
    
    temp_data.choices = data.choices(ind,:);
    temp_data.rewards = data.rewards(ind,:);
    nll_opt  = @(theta) nll(temp_data, theta);

    [optimizer, value] = fminunc(nll_opt, theta_zero);

    individuals(ind,:) = [optimizer, value];
    
end

%Creating a Table for optimized parameters 
Learning_Rate = individuals(:,1);
Inverse_Temp = individuals(:,2);
Participants = 1:1:size(data.choices, 1);
Participants = Participants';
Table1 = table(Participants, Learning_Rate, Inverse_Temp);
writetable(Table1,'Param_opt_zero.txt');

%Start trying multiple initalizations

eps_multiple = [-1.5 -1 -0.5 0 0.2 0.5 1 1.5 2];
beta_multiple = [0 3 5 8 12 15 -5 -10 20];

for ind_eps = 1:size(eps_multiple, 2)
    
    for ind_beta = 1:size(beta_multiple, 2)
        
        for ind = 1: size(data.choices,1) 
            temp_data.choices = data.choices(ind,:);
            temp_data.rewards = data.rewards(ind,:);
            nll_opt  = @(theta) nll(temp_data, [eps_multiple(ind_eps) beta_multiple(ind_beta)]);

            [optimizer, value] = fminunc(nll_opt, theta_zero);

            individuals_multi(ind,:) = [optimizer, value];
        end
        
    end
    
end

% End trying multiple initializations
% 
% Start visualizing the fitted parameters

figure(5)
plot(1:23, logsig(individuals(1:23,1)),'r');
hold on
plot(24:48, logsig(individuals(24:48,1)),'g');
hold on
plot(1:23, individuals(1:23,2),'r');
plot(24:48, individuals(24:48,2),'g');
hold off
title('Visualizing the fitted Parameters');
xlabel('Participants Index');
ylabel('Learning Rates and Inverse Temperatures');

%End visualizing the fitted parameters (check)

%Start Computing Correlations

%within groups
[R_pat, P_pat] = corrcoef(individuals(1:23,1), individuals(1:23,2));
disp('correlation for patients is:'); disp(R_pat);

[R_heal P_heal] = corrcoef(individuals(24:48,1), individuals(24:48,2));
disp('correlation for healthy is:'); disp(R_heal);

%Across gorups
[R_all P_all] = corrcoef(individuals(:,1), individuals(:,2));
disp('correlation across groups is:'); disp(R_all);

%End Computing Correlations


%% Task(e) Gradient

for ind = 1: size(data.choices, 1)
    
    temp_data.choices = data.choices(ind,:);
    temp_data.rewards = data.rewards(ind,:);
    nll_opt_grad  = @(theta) nll_withgradient(temp_data, theta_zero);

    options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
    [optimizer, value] = fminunc(nll_opt_grad, theta_zero, options);

    individuals_grad(ind,:) = [optimizer, value];
    
end

% Visualize gradient results

% Start visualizing the fitted parameters

figure(5)
plot(1:23, logsig(individuals_grad(1:23,1)),'r');
hold on
plot(24:48, logsig(individuals_grad(24:48,1)),'g');
hold on
plot(1:23, individuals_grad(1:23,2),'r');
plot(24:48, individuals_grad(24:48,2),'g');
hold off
title('Visualizing the fitted Parameters');
xlabel('Participants Index');
ylabel('Learning Rates and Inverse Temperatures');

% End visualizing the fitted parameters (check)

%% Task(f) Group Comparision

[h1,p1,ci1,stats1] = ttest2(individuals(1:23,1),individuals(24:48,1)); % learning rate
disp('The null hypothesis, p value and t-stats for learning rate is:'); disp(h1); disp(p1); disp(stats1);

[h2,p2,ci2,stats2] = ttest2(individuals(1:23,2),individuals(24:48,2)); % Inverse temp
disp('The null hypothesis, p value and t-stats for inverse temp is:'); disp(h2); disp(p2); disp(stats2);
%% Task(g) Parameter Recovery

% calculating mean and variance

%learning rate
mean_eps = mean(individuals(:,1)); 
save mean_eps.mat mean_eps
var_eps = var(individuals(:,1));
save var_eps.mat var_eps
disp('The mean and variance for learning rate is:');disp([mean_eps var_eps]);

% Inverse temp
mean_beta = mean(individuals(:,2));
save mean_beta.mat mean_beta
var_beta = var(individuals(:,2));
save var_beta.mat var_beta
disp('The mean and variance for inverse temp is:');disp([mean_beta var_beta]);

% Sampling X from the distribution of fitted parameters
points = 48; % no. of points sampled

for x = 1: 10
    
new_eps_vec = var_eps*randn(points,1) + mean_eps;
new_beta_vec = var_beta*randn(points,1) + mean_beta;

% Visualize
figure(6)
plot(logsig(new_eps_vec));
hold on
plot(new_beta_vec);
title('Sampled Parameter values');
xlabel('Index');
ylabel('Learning Rates and Inverse temperatures');

end
hold off


% Starting Simulation to create choices and rewards

for i = 1: size(new_beta_vec,1)
    new_theta =[new_eps_vec(i) new_beta_vec(i)];
    new_data = simulations(new_theta, sim_trials);
    new_reward(i,:) = new_data.rewards;
    new_choices(i,:) = new_data.choices;
end

% Fitting the parameters computing NLL

for ind=1:size(new_reward,1)
    
temp_data.choices = new_choices(ind,:);
temp_data.rewards = new_reward(ind,:);
nll_opt  = @(theta) nll(temp_data, theta);

theta = [0, 0];
[new_optimizer, new_value] = fminunc(nll_opt, theta);

new_individuals(ind,:) = [new_optimizer, new_value];
end

% Reporting Correlation b/w new fitted values and the ones used to sample

[R_newsample_eps P_newasmple_eps] = corrcoef(new_eps_vec, new_individuals(:,1));
disp('correlation across groups is:'); disp(R_newsample_eps);

[R_newsample_beta P_newsample_beta] = corrcoef(new_beta_vec, new_individuals(:,2));
disp('correlation across groups is:'); disp(R_newsample_beta);

% Visualizing these correlations
figure(7)
corrplot([new_eps_vec new_individuals(:,1)]);
%title('Pearson cofficients of Correltaion Matrix (Learning Rate)');

figure(8)
corrplot([new_beta_vec new_individuals(:,2)]);
%title('Pearson cofficients of Correltaion Matrix (Inverse Temp)');


% Exploring sampleX size and simulation trials
points_exp = 50:10:500;
sim_size = 100:20:1000;
for index = 1: size(sim_size, 2)
    
        exp_eps_vec = var_eps*randn(points,1) + mean_eps;
        exp_beta_vec = var_beta*randn(points,1) + mean_beta;
        
    %for sim_size = 100:10:1000
        new_reward=[];
        new_choices= [];
        
        for i = 1: size(exp_beta_vec,1)
            new_theta =[exp_eps_vec(i) exp_beta_vec(i)];
            new_data = simulations(new_theta, sim_size(index));
            new_reward(i,:) = new_data.rewards;
            new_choices(i,:) = new_data.choices;
        end

        newexp_individuals = zeros(size(new_reward,1),3);

        for ind=1:size(new_reward,1)
    
            temp_data.choices = new_choices(ind,:);
            temp_data.rewards = new_reward(ind,:);
            nll_opt  = @(theta) nll(temp_data, theta);

            theta = [0, 0];
            [new_optimizer, new_value] = fminunc(nll_opt, theta);

            newexp_individuals(ind,:) = [new_optimizer, new_value];
        end
        
%         [hexp1,pexp1,ciexp1,statsexp1] = ttest2(individuals(:,1), newexp_individuals(:,1));
%         [hexp2,pexp2,ciexp2,statsexp2] = ttest2(individuals(:,2), newexp_individuals(:,2));
%         t_value1(index) = statsexp1.tstat;
%         t_value2(index) = statsexp2.tstat;
        [R_temp1 P_temp1] = corrcoef(exp_eps_vec, newexp_individuals(:,1));
        [R_temp2 P_temp2] = corrcoef(exp_beta_vec, newexp_individuals(:,2));
        coef_eps(index) = R_temp1(1,2);
        coef_beta(index) = R_temp2(1,2);
    %end 
end
figure(16)
plot(points_exp,coef_eps,'r');
hold on
plot(points_exp, coef_beta,'g');
hold off
title('Peformance of Recovery');
xlabel('Simulation Trials');
ylabel('tstat from two sample t-test');

%% Task(h) Alternative Models

% Model 2

model_2 = simulations_2(theta_task_a, sim_trials); % the function

%start plotting figures
figure(9)
plot(model_2.values_A);
title('Evolution of Values for stimulus A and B(Model 2');
xlabel('Trial Number');
ylabel('Value associted with A and B');
hold on
plot(model_2.values_B);
hold off

figure(10)
val_diff = (model_2.values_A - model_2.values_B);
plot(val_diff);
title('Evolution of Difference of Values for A nd B (Model 2)');
xlabel('Trial Number');
ylabel('Difference Value or V(A) - V(B)');
% end plotting figures

% Computing NLL
nll_participant_2_model_2 = nll_2(data_part_2, [0.4 5]);

disp('The NLL for Participant 2 in model 2 is:');
disp(nll_participant_2_model_2);

% Fitting the model

for ind = 1: size(data.choices, 1)
    
    temp_data.choices = data.choices(ind,:);
    temp_data.rewards = data.rewards(ind,:);
    nll_opt  = @(theta) nll_2(temp_data, theta);

    [optimizer, value] = fminunc(nll_opt, theta_zero);

    individuals_model2(ind,:) = [optimizer, value];
    
end

% Visualizing the fitted parameters
figure(11)
plot(1:23, logsig(individuals_model2(1:23,1)),'r');
hold on
plot(24:48, logsig(individuals_model2(24:48,1)),'g');
hold on
plot(1:23, individuals_model2(1:23,2),'r');
plot(24:48, individuals_model2(24:48,2),'g');
hold off
title('Visualizing the fitted Parameters(Model 2)');
xlabel('Participants Index');
ylabel('Learning Rates and Inverse Temperatures');


% Model 3

model_3 = simulations_3(theta_task_a, sim_trials); % the function

%start plotting figures
figure(12)
plot(model_3.values_A);
title('Evolution of Values for stimulus A and B(Model 3');
xlabel('Trial Number');
ylabel('Value associted with A and B');
hold on
plot(model_3.values_B);
hold off

figure(13)
val_diff = (model_3.values_A - model_3.values_B);
plot(val_diff);
title('Evolution of Difference of Values for A nd B (Model 3)');
xlabel('Trial Number');
ylabel('Difference Value or V(A) - V(B)');
% end plotting figures

% Computing NLL
nll_participant_2_model_3 = nll_3(data_part_2, [0.4 5]);

disp('The NLL for Participant 2 in model 2 is:');
disp(nll_participant_2_model_3);

% Fitting the model

for ind = 1: size(data.choices, 1)
    
    temp_data.choices = data.choices(ind,:);
    temp_data.rewards = data.rewards(ind,:);
    nll_opt  = @(theta) nll_3(temp_data, theta);

    [optimizer, value] = fminunc(nll_opt, theta_zero);

    individuals_model3(ind,:) = [optimizer, value];
    
end

% Visualizing the fitted parameters
figure(14)
plot(1:23, logsig(individuals_model3(1:23,1)),'r');
hold on
plot(24:48, logsig(individuals_model3(24:48,1)),'g');
hold on
plot(1:23, individuals_model3(1:23,2),'r');
plot(24:48, individuals_model3(24:48,2),'g');
hold off
title('Visualizing the fitted Parameters(Model 3)');
xlabel('Participants Index');
ylabel('Learning Rates and Inverse Temperatures');

%% Task(i) Model Comparision

% Computing AIC
p = 2; % parameter count for all models
AIC_model(1) = sum((2*individuals(:,3)) + (2*p));
AIC_model(2) = sum((2*individuals_model2(:,3)) + (2*p));
AIC_model(3) = sum((2*individuals_model3(:,3)) + (2*p));

% Computing BIC
n = 250; % no. of observations

BIC_model(1) = sum((2*individuals(:,3)) + (log(n)*p));
BIC_model(2) = sum((2*individuals_model2(:,3)) + (log(n)*p));
BIC_model(3) = sum((2*individuals_model3(:,3)) + (log(n)*p));

% Reporting results

disp('The AIC values are(model as index):'); disp(AIC_model);
disp('The BIC values are(model as index):'); disp(BIC_model);

figure(15)
plot(AIC_model);
hold on 
plot(BIC_model);
title('AIC and BIC values');
xlabel('Model as Index');
ylabel('AIC & BIC values');

%% Task(j) Model recovery and Confusion Matrix 
% simulate multiple data

for i= 1:100

    for j = 1:3
        
    % first model fit
        for ind = 1:48
            
            if j == 1
            datamodel = simulations(theta_task_a, sim_trials);
            end
            if j == 2
            datamodel = simulations_2(theta_task_a, sim_trials);
            end
            if j == 3
            datamodel = simulations_3(theta_task_a, sim_trials);
            end
            
            temp_data.choices = datamodel.choices;
            temp_data.rewards = datamodel.rewards;
    
            nll_opt1  = @(theta) nll(temp_data, theta); % first
            nll_opt2  = @(theta) nll_2(temp_data, theta); % second
            nll_opt3  = @(theta) nll_3(temp_data, theta); % third
        
            [optimizer1, value1] = fminunc(nll_opt1, theta_zero); % first
            individuals_model1_conf(ind,:) = [optimizer1, value1];
        
            [optimizer2, value2] = fminunc(nll_opt2, theta_zero); % second
            individuals_model2_conf(ind,:) = [optimizer2, value2];
        
            [optimizer3, value3] = fminunc(nll_opt3, theta_zero); % third
            individuals_model3_conf(ind,:) = [optimizer3, value3];
    
        end
        
        AIC_model_conf(1) = sum((2*individuals_model1_conf(:,3)) + (2*p));
        AIC_model_conf(2) = sum((2*individuals_model2_conf(:,3)) + (2*p));
        AIC_model_conf(3) = sum((2*individuals_model3_conf(:,3)) + (2*p));
        
        BIC_model_conf(1) = sum((2*individuals_model1_conf(:,3)) + (log(n)*p));
        BIC_model_conf(2) = sum((2*individuals_model2_conf(:,3)) + (log(n)*p));
        BIC_model_conf(3) = sum((2*individuals_model3_conf(:,3)) + (log(n)*p));
        
        compare(j,:,i) = [BIC_model_conf];
        
    end

end

% Confusion Matrix AIC

targets = eye(3);

for k = 1:100
    
    AIC_min = min(compare(:,1:3,k),[],2);
    [x, y] = find(compare(:,1:3,k) == AIC_min);
    
    for run = 1: size(y,1)
        
        compare(x(run),y(run),k) = 1;
        temp = compare(:,1:3,k);
        temp(temp ~= 1) = 0;
        compare(:,1:3,k) = temp;
        
    end

 [c,mat(:,:,k),indexed,per] = confusion(targets, compare(:,1:3,k));   
end

confused = sum(mat, 3)./100;

