% Sparse equalizer
clear all
close all
clc
training_len = 10^4; % length of the training sequence
equalizer_len = 200; % length of the equalizer
chan_len = 50;
data_len = 10^6; % length of the data sequence

% SNR parameters
SNR_dB = 15; % SNR (dB)
noise_var = 1/(2*10^(0.1*SNR_dB)); % noise variance

% source
a = randi([0 1],1,training_len);

% bpsk mapper
training_seq = 1-2*a;

% fade chan
fade_chan = zeros(1,chan_len);
fade_chan([1 10 20 30 40 50]) = [0.9 0.1 0.1 0.1 0.1 0.1];
fade_chan = fade_chan/norm(fade_chan);

% noise 
noise = normrnd(0,sqrt(noise_var),1,training_len+chan_len-1);

% channel output
chan_op = conv(fade_chan,training_seq)+noise;

% --------- RECEIVER----------------------------------
% autocorrelation of the output sequence
auto_corr_vec = xcorr(chan_op,chan_op,'unbiased');
mid_point = (length(auto_corr_vec)+1)/2;
c = auto_corr_vec(mid_point:mid_point+equalizer_len-1); % first column of toeplitz 
r = fliplr(auto_corr_vec(mid_point-equalizer_len+1:mid_point));
Rvv_Matrix = toeplitz(c,r);

% Cholesky decomposition
R = chol(Rvv_Matrix);

% cross correlation 
cross_corr_vec = xcorr(chan_op(1:length(training_seq)),training_seq,'unbiased');
mid_point = (length(cross_corr_vec)+1)/2;
cross_corr_vec = cross_corr_vec(mid_point:mid_point+equalizer_len-1).';

% OMP algorithm
sensing_matrix = R;
measurement_vec = transpose(inv(R))*cross_corr_vec;

index = [];
[dummy,index(1)] = max(sensing_matrix'*measurement_vec);
A = sensing_matrix(:,index);
x = A\measurement_vec; %least squares solution
residue = [];
residue(:,1) = A*x -measurement_vec;

for i1= 2:50 % only 50 taps
[dummy,index(i1)] = max(sensing_matrix'*residue(:,i1-1));
A = [A sensing_matrix(:,index(i1))];
x = A\measurement_vec;
residue(:,i1) = A*x -measurement_vec;
end
len = length(x);
x = [x;zeros(equalizer_len-len,1)];
equalizer = zeros(equalizer_len,1);
for i1 = 1:length(index)
    equalizer(index(i1)) = x(i1);
end
equalizer = equalizer.'; % now a row vector

%------------------ data transmission phase----------------------------
% source
data_a = randi([0 1],1,data_len);

% bpsk mapper (bit '0' maps to 1 and bit '1' maps to -1)
data_seq = 1-2*data_a;

% AWGN
noise = normrnd(0,sqrt(noise_var),1,data_len+chan_len-1);

% channel output
chan_op = conv(fade_chan,data_seq)+noise;

% equalization
equalizer_op = conv(chan_op,equalizer);
equalizer_op = equalizer_op(1:data_len);

% demapping symbols back to bits
dec_a = equalizer_op<0;

% bit error rate
ber = nnz(dec_a-data_a)/data_len

