clc;
clear all;
close all;
%s = 0:0.01:0.30; % changing the variance
%v = -1.0:0.01:1.0; % changing drift velocity
%z = 0:0.001:0.10; % initial value
%a = 0:0.01:1.0; % decision seperation distance

%for iteration = 1: size(a,2)

t1 = 1;
t2 = 1;
for i = 1:10000

%% Part 1. Drift Diffusion Process
% Task (a)

% Initial Simulations

v = 0.19; % the mean drift rate
a = 0.1; % separation between boundaries
s = 0.085; % variance
z = a/2; % starting point
dt = 0.001; % time step

% starting the simulation
W=[];
W(1) = z; % Weiner diffusion process initial value 
t_elapse = 0; % initialzing the time 
t =1 ; % initializing index for diffusion storage
while W(t)>=0 && W(t)<=a
    
    W(t+1) = W(t) + v*dt + s*randn()*sqrt(0.001);
    t = t+1;
end

t_elapsed = (t-1)*dt; % time taken for the decision

if W(t)>a
    W(t) = a;
    %disp("The decision is h+");

else
    W(t) = 0;
    %disp("The decision is h-");
end

% plot(0:dt:t_elapsed, W,'LineWidth',1);
% title('Weiner Diffusion Process');
% xlabel('Time Elapsed (sec)');
% ylabel('Decision Distance with Boundaries');
% hold on;
%disp(W(t)); % print the difussion process value
%disp("The time elapsed is :")
%disp(t_elapsed);

store_W(i) = W(t);
if W(t)== a
    res_time_cor(t1) = t_elapsed;
    t1= t1+1;
end

if W(t) == 0
    res_time_incor(t2) = t_elapsed;
    t2= t2+1;
end

end

accuracy = (size(store_W(store_W==a),2)/i)*100; % percentage
avg_res1 = mean(res_time_cor);
avg_res2 = mean(res_time_incor);
var_res1 = var(res_time_cor);
var_res2 = var(res_time_incor);
figure(1)
histogram(res_time_cor);
title('Distribution of Response Time Correct Hypothesis')
ylabel('Counts');
xlabel('Response Times')
figure(2)
histogram(res_time_incor);
title('Distribution of Response Time Incorrect Hypothesis')
ylabel('Counts');
xlabel('Response Times')

%end
accuracy
avg_res1
avg_res2
var_res1
var_res2
figure(3)
plot(a, accuracy);
title('Average Accuracy with Varying Decision Seperation Distance');
xlabel('Decision Seperation Distance');
ylabel('Accuracy (%)');

figure(4)
plot(a, avg_res);
title('Average Response Time with Varying Decision Seperation Distance');
xlabel('Decision Seperation Distance');
ylabel('Response Time');


    