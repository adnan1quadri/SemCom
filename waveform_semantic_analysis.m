clear;
clc;
close all;
addpath 'C:\Users\User\Documents\MATLAB\Examples\R2024b\deeplearning_shared\SpectrumSensingRadarCommDeepLearningExample'

%% RADAR Waveform parameters: airport surveillance radar
% setup waveform
fc=2.8e9;                   % center frequency [Hz]
fs=61.44e6;                 % sampling frequency [Hz]
prf=fs/ceil(fs/1050);       % pulse repetition rate [Hz]
pulseWidth=1e-06;           % pulsewdith [s]
wav = phased.RectangularWaveform('SampleRate',fs,'PulseWidth',pulseWidth,'PRF',prf,'NumPulses',3);

% setup antenna
rad=2.5;            % radius [m]
flen=2.5;           % focal length [m]
antele=design(reflectorParabolic('Exciter',horn),fc);
antele.Exciter.Tilt=90;
antele.Exciter.TiltAxis=[0 1 0];
antele.Tilt=90;
antele.TiltAxis=[0 1 0];
antele.Radius=rad;
antele.FocalLength=flen;
ant=phased.ConformalArray('Element',antele);

% setup transmitter and receiver
power_ASR=25000;    % transmit power [W]
gain_ASR=32.8;      % transmit gain [dB]
radartx = phased.Transmitter('PeakPower',power_ASR,'Gain',gain_ASR);

%% Radar, 5G, and LTE Tx/RX positioning 
% Randomly placed in 2kmx2km region
rxpos_horiz_minmax=[-1000 1000];
rxpos_vert_minmax=[0 2000];


% define radar position
radartxpos=[0 0 15]';
radartxaxes=rotz(0);
radarchan=phased.ScatteringMIMOChannel('TransmitArray',ant,... 
    'ReceiveArray',phased.ConformalArray('Element',phased.IsotropicAntennaElement),...
    'CarrierFrequency',fc,...
    'SpecifyAtmosphere',true,...
    'SampleRate',fs,...
    'SimulateDirectPath',false,...
    'MaximumDelaySource','Property',...
    'MaximumDelay',1,...
    'TransmitArrayMotionSource','Property',...
    'TransmitArrayPosition',radartxpos,...
    'TransmitArrayOrientationAxes',radartxaxes,...
    'ReceiveArrayMotionSource','Input port',...
    'ScattererSpecificationSource','Input port');

% define wireless transmitter position
commtxpos=[200 0 450]';
commtxaxes=rotz(0);
commchan=phased.ScatteringMIMOChannel('TransmitArray',phased.ConformalArray('Taper',10),... 
    'ReceiveArray',phased.ConformalArray('Element',phased.IsotropicAntennaElement),...
    'CarrierFrequency',fc,...
    'SpecifyAtmosphere',true,...
    'SampleRate',fs,...
    'SimulateDirectPath',false,...
    'MaximumDelaySource','Property',...
    'MaximumDelay',1,...
    'TransmitArrayMotionSource','Property',...
    'TransmitArrayPosition',commtxpos,...
    'TransmitArrayOrientationAxes',commtxaxes,...
    'ReceiveArrayMotionSource','Input port',...
    'ScattererSpecificationSource','Input port');
%% Generate Training Data: Noise, LTE, 5G NR, RADAR--------------
% Number of data generated
numTrainingData=500;
imageSize=[256 256];

% Define data directory
classNames = ["Noise" "LTE" "NR" "Radar"];
pixelLabelID = [0 1 2 3];
useDownloadedDataSet = true;
saveFolder = tempdir; 
trainingFolder=fullfile(saveFolder,'RadarCommTrainData');
if ~useDownloadedDataSet
    % Generate data
    helperGenerateRadarCommData(fs,wav,radartx,radarchan,commchan,rxpos_horiz_minmax,rxpos_vert_minmax,numTrainingData,trainingFolder,imageSize);
else
    dataURL = 'https://ssd.mathworks.com/supportfiles/phased/data/RadarCommSpectrumSensingData.zip';
    zipFile = fullfile(saveFolder,'RadarCommSpectrumSensingData.zip');
    if ~exist(zipFile,'file')
        websave(zipFile,dataURL);
        % Unzip the data
        unzip(zipFile,saveFolder)
    end
end 

% Data
trainingFolder = fullfile(saveFolder,'RadarCommTrainData');
imds = imageDatastore(trainingFolder,'IncludeSubfolders',false,'FileExtensions','.png');

% Label
pxdsTruth = pixelLabelDatastore(trainingFolder,classNames,pixelLabelID,...
  'IncludeSubfolders',false,'FileExtensions','.hdf');

% only to analyze datasets statistics-----------------------
tbl = countEachLabel(pxdsTruth); frequency = tbl.PixelCount/sum(tbl.PixelCount);
figure;bar(1:numel(classNames),frequency);grid on;xticks(1:numel(classNames)) 
xticklabels(tbl.Name);xtickangle(45);ylabel('Frequency');

%% Prepare Training, Validation and Test Sets-----------------
% 80/20 train/validation set partitioning--------------
[imdsTrain,pxdsTrain,imdsVal,pxdsVal] = helperRadarCommPartitionData(imds,pxdsTruth,[80 20]);
cdsTrain = pixelLabelImageDatastore(imdsTrain,pxdsTrain,'OutputSize',imageSize);
cdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal,'OutputSize',imageSize);

%% Load ResNet model to initialize DNN training
baseNetwork = 'resnet50';
lgraph = deeplabv3plus(imageSize,numel(classNames),baseNetwork);

% Address Class Imbalance (i.e prepare balanced dataset for training)
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;
classWeights = classWeights/(sum(classWeights)+eps(class(classWeights)));

% Configure training options (DL toolbox)
opts = trainingOptions("sgdm",...
  MiniBatchSize = 40,...
  MaxEpochs = 40, ...
  LearnRateSchedule = "piecewise",...
  InitialLearnRate = 0.02,...
  LearnRateDropPeriod = 10,...
  LearnRateDropFactor = 0.1,...
  ValidationData = cdsVal,...
  ValidationPatience = 5,...
  Shuffle="every-epoch",...
  Plots = "training-progress",...
  OutputNetwork = "best-validation-loss",...
  Metrics = "accuracy",...
  ExecutionEnvironment = "auto");

%% ******************** IMPORTANT SETTING *************************
% CHANGE trainNow to TRUE to start training. Otherwise, will load
% pretrained version ----------------------------------------------
trainNow = false;
if trainNow
  [net,trainInfo] = trainnet(cdsTrain,lgraph,...
       @(ypred,ytrue) helperWeightedCrossEntropyLoss(ypred,ytrue,classWeights),opts);  %#ok<UNRCH> 
else
  load(fullfile(saveFolder,'helperRadarCommTrainedNet.mat'),'net');
end

%% Synthesize Test Signal to evaluate network performance --------
% Generate testing data
numTestData=100;
testFolder = fullfile(saveFolder,'RadarCommTestData');
if ~useDownloadedDataSet
    helperGenerateRadarCommData(fs,wav,radartx,radarchan,commchan,rxpos_horiz_minmax,rxpos_vert_minmax,numTestData,testFolder,imageSize);
end

% Prep and test the testing data
imds = imageDatastore(testFolder,'IncludeSubfolders',false,'FileExtensions','.png');
pxdsResults = semanticseg(imds,net,"WriteLocation",saveFolder, ...
    'Classes',classNames,'ExecutionEnvironment','auto');

% Prep the ground truth testing labels
pxdsTruth = pixelLabelDatastore(testFolder,classNames,pixelLabelID,...
  'IncludeSubfolders',false,'FileExtensions','.hdf');
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

% Test Set semantic segmentation analytics
cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
  classNames, Normalization='row-normalized');
cm.Title = 'Normalized Confusion Matrix';

imageIoU = metrics.ImageMetrics.MeanIoU;figure;histogram(imageIoU);grid on;xlabel('IoU');ylabel('Number of Frames');title('Frame Mean IoU');

%% Performance Evalaution: pass along specttogram 142 to see network classifications and comapre with ground truth
imgIdx = 142;
rcvdSpectrogram = readimage(imds,imgIdx);
trueLabels = readimage(pxdsTruth,imgIdx);
predictedLabels = readimage(pxdsResults,imgIdx);
figure
helperSpecSenseDisplayResults(rcvdSpectrogram,trueLabels,predictedLabels, ...
  classNames,fs,0,1/prf)
