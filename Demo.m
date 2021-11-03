clear all
clc
close all
dataname='exampledata.mat';  %% directionary of data, for example, 'C:\**\**\exampledata.mat';
%% in the dataset 'exampledata.mat'
% DTra1 - training data;
% LTra1 - class labels of training data;
% DTes1 - testing data;
% LTes1 - class labels of testing data;
%%
NumBaseLearner=20;      %% number of base classifiers used for creating the ensemble;
GranLevel=12;           %% level of granularity, G has to be a positive integer.
[PredFAS,AccFAS]=FWAdaBoostSOFIES(dataname,NumBaseLearner,GranLevel);
PredFAS %% predicted class labels of testing data; 
AccFAS  %% testing accuracy.