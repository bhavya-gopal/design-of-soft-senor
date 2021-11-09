%% Design of Soft Sensor: Residue Prediction


%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('dist_data.mat');
X_train = data(1:700, 1:12);
y_train = data(1:700, 14);

X_test = data(700:end, 1:12);
y_test = data(700:end, 14);
m = length(y_train);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X_train mu sigma] = featureNormalize(X_train);

% Add intercept term to X
X_train = [ones(m, 1) X_train];

%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 200;

% Initial Theta and Run Gradient Descent 
theta = zeros(13, 1);
[theta, J_history] = gradientDescentMulti(X_train, y_train, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Predict the xD value using gradient descent 
X_test_norm= featureNormalize(X_test);
[mt nt] = size(X_test);

% Add intercept term to X_test
X_test_norm = [ones(mt, 1) X_test_norm];
xR1 = 0; 
xR1= X_test_norm*theta;


fprintf(['Predicted xR value ' ...
         '(using gradient descent):\n %f\n'], xR1);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

%% Normal Equations
% Calculate the parameters from the normal equation
theta = normalEqn(X_train, y_train);
% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the xR value using Normal Equation
% ====================== YOUR CODE HERE ======================
xR2 = 0; 
xR2= X_test_norm*theta;
% ============================================================
fprintf(['Predicted xR value' ...
         '(using normal equations):\n %f\n'], xR2);
% ============================================================
% Plot to find accuracy of Machine Learning Algorithms
plot(y_test)
hold on
plot(xR1)
plot(xR2)
hold off
xlabel('Trials')
ylabel('Residue')
legend({'Experimental Data', 'Gradient Descent', 'Normal Equation'},'Location', 'southeast')
title('Accuracy of Different Algorithms for Residue Prediction')