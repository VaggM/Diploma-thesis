% Implementation of the discriminative alpha-beta divergence for positive definite matrices
%  if you plan to use this code, please cite 
%
%  A. Cherian, P. Stanitsas, M. Harandi, V. Morellas, and N. Papanikolopoulos, Learning Discriminative Alpha-Beta Divergence for Positive Definite Matrices, ICCV, 2017.
%
% This code is released on the BSD3 license.
% This code should not be used for non-commercial purposes. 
% The authors are not liable to any loss or damage caused by running this code.
% For any issues/bugs, please contact anoop.cherian@gmail.com.

% Copyright (c) 2017, Panagiotis Stanitsas and Anoop Cherian
% All rights reserved.
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
% 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
% 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
% PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
% INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
% OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
% OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\


clear all
clc
%% Add Manopt in the filepath
% cd manopt
% run importmanopt.m
% cd ..
% %% Add support functions
% addpath('Support_Functions')
%% Load data
load demo_dataset.mat
%% Initialize parameters
params.iter = 3;
params.num_atoms_per_class = 1;
params.lam = 10;
params.loss_func = @RRL;
params.loss_func_gradV = @RRL_gradV;
params.loss_param_initialize = @RRL_param_initialize;
params.loss_param_update = @RRL_param_update;
params.loss_class_accuracy = @RRL_class_accuracy;

%% Run IDDL
fprintf('\n\n')
fprintf('Running IDDL...\n')
fprintf('------------------------------------------------------------------\n')
[BB, ab, loss_params, V] = IDDL(X_train, X_test, train_labels, test_labels, params);



