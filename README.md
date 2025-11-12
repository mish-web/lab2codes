# lab2codes
Codes
% 1. Set Parameters
n = 15;
p = 0.25;

% 2. Generate Adjacency Matrix
A = zeros(n, n); 
for i = 1:n
    for j = i+1:n % Iterate over unique pairs (i, j) where i < j
        % Generate a random number R, if R <= p, an edge exists
        if rand() <= p
            A(i, j) = 1;
            A(j, i) = 1; % Undirected graph
        end
    end
end

% 3. Create and Visualize the Graph
G = graph(A);
figure;
plot(G, 'NodeLabel', 1:n, 'Layout', 'force');
title(sprintf('Erdos-Renyi Graph G(%d, %.2f)', n, p));

% 4. Analyze Properties
num_edges = numedges(G);
degrees = degree(G);
avg_degree = mean(degrees);
is_connected = all(conncomp(G) == 1);

fprintf('Generated Edges: %d (Expected: %.2f)\n', num_edges, n*(n-1)/2*p);
fprintf('Average Degree: %.2f (Expected: %.2f)\n', avg_degree, (n-1)*p);
fprintf('Is Connected: %s\n', string(is_connected));% 1. Set parameters for a large sequence 
N_trials = 1000; % Changed to 1000 trials
p_success = 0.25;

% 2. Generate Bernoulli Sequence
% rand(1, N_trials) generates N_trials random numbers between 0 and 1.
% (rand() <= p_success) evaluates to 1 (true/success) if the random number 
% is less than or equal to p, and 0 (false/failure) otherwise.
sequence = (rand(1, N_trials) <= p_success); 

% 3. Plot Histogram of Successes (Ones)
figure;
% The histogram function automatically counts the occurrences of 0s and 1s.
histogram(sequence, 'Normalization', 'probability');
title(sprintf('Histogram of Outcomes for Bernoulli Sequence (N=1000, p=%.2f)', p_success));
xlabel('Value (0 = Failure, 1 = Success)');
ylabel('Probability/Relative Frequency');

% Ensure the x-axis ticks are clearly labeled for 0 and 1
xticks([0 1]);
xticklabels({'0 (1-p)', '1 (p)'});

% MATLAB Code for Average Bits per Bit vs. Block Size
clear; close all; clc;
% --- 1. System Parameters (Updated) ---
p_edge = 0.25;       % Edge connection probability (p)
N_bernoulli = 1000;  % Total length of the Bernoulli sequence (N)
block_sizes = [1, 2, 4, 8]; % Block sizes (B) for Huffman coding analysis
% --- 2. Bernoulli Sequence Generation and Theoretical Entropy ---
% Generate a long Bernoulli sequence
bernoulli_seq = rand(1, N_bernoulli) < p_edge; % 0s and 1s
% Calculate Source Entropy (H(p)) - The ultimate theoretical limit
H_p = -p_edge * log2(p_edge) - (1 - p_edge) * log2(1 - p_edge);
fprintf('Theoretical Source Entropy H(X): %.4f bits/symbol\n', H_p);
% --- 3. Huffman Coding Analysis for Different Block Sizes ---
L_avg_results = zeros(size(block_sizes)); % Array to store H(X^B)/B
fprintf('\n--- Huffman Coding Block Analysis ---\n');
for i = 1:length(block_sizes)
B = block_sizes(i);
fprintf('Processing Block Size B = %d...\n', B);
% a. Block the sequence
% Ensure sequence length is a multiple of B (truncate if necessary)
L = floor(length(bernoulli_seq) / B) * B;
blocked_seq = reshape(bernoulli_seq(1:L), B, L/B)';
% b. Convert binary blocks to decimal symbols
M = 2^B; % Number of possible symbols
% MATLAB command to convert binary rows (MSB first) to decimal symbols
symbols = bi2de(blocked_seq);
% c. Calculate Symbol Frequencies and Probabilities
counts = histcounts(symbols, 0:M);
% Normalize counts to get probabilities, excluding zero counts (symbols not observed)
probabilities = counts / sum(counts);
probabilities(probabilities == 0) = []; % Remove 0 probabilities to avoid log2(0)
% d. Calculate Block Entropy (H(X^B))
H_B = -sum(probabilities .* log2(probabilities));
% e. Calculate Average Bits per Source Bit (Approximation: L_avg/B ~ H(X^B)/B)
L_avg_per_bit = H_B / B;
% f. Store Results
L_avg_results(i) = L_avg_per_bit;
fprintf('  Block Entropy H(X^B): %.4f bits/symbol\n', H_B);
fprintf('  Average Bits per Bit (L_avg/B): %.4f bits/bit\n', L_avg_per_bit);
end
% --- 4. Plotting the Results ---
figure('Name', 'Bernoulli Sequence: Bits per Bit vs. Block Size');
plot(block_sizes, L_avg_results, '-o', 'LineWidth', 2);
hold on;
% Plot the theoretical limit (H(X)) as a horizontal line
plot(block_sizes, ones(size(block_sizes)) * H_p, '--r', 'DisplayName', sprintf('Entropy Limit H(X) = %.4f', H_p));
hold off;
title('Bernoulli Sequence: Average Bits per Source Bit vs. Block Size');
xlabel('Block Size (bits)');
ylabel('Average Bits per Source Bit (L_{avg}/B)');
legend('Empirical H(X^B)/B', 'Location', 'NorthEast');
grid on;
% Save the figure
saveas(gcf, 'Bernoulli sequence_ bits per bit vs block size.png');
%% Plot 2: Top 20 Most Frequent ER Graphs
clear; clc;

n = 15;
p = 0.25;
N_samples = 500000;  % Large sample for stable ranking

M = nchoosek(n,2);
graph_hash = zeros(N_samples, 1, 'uint64');

rng(123);
for i = 1:N_samples
    adj = rand(M,1) < p;
    % Convert bitvector to 64-bit hash (M=105 > 64 → use two uint64)
    hash1 = sum(uint64(adj(1:64)) .* (2.^(0:63))');
    hash2 = sum(uint64(adj(65:end)) .* (2.^(0:40))');
    graph_hash(i) = hash1;  % Simplified: use first 64 bits
end

[,,ic] = unique(graph_hash);
freq = accumarray(ic, 1);
[~, sortIdx] = sort(-freq);
top20_freq = freq(sortIdx(1:20));

figure('Color','k');
bar(1:20, top20_freq, 'FaceColor', [0.1 0.7 0.9], 'EdgeColor','w');
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');
xlabel('Graph rank (1-20)', 'Color','w');
ylabel('Frequency', 'Color','w');
title('Top 20 most frequent ER graphs (n=15, p=0.25)', 'Color','w');
ylim([0 200]); grid on; set(gca,'GridColor',[0.3 0.3 0.3]);
%% MATLAB Code: Codeword Length Histogram for 15 Symbols

% --- 1. Define Data (Total number of symbols = 15) ---
% This vector contains the codeword length for each of the 15 symbols.
% The sum of the frequencies in the histogram will equal 15.
codeword_lengths = [
    10, 10, 10,       ... % 3 symbols have length 10
    11, 11, 11, 11, 11, ... % 5 symbols have length 11
    12, 12, 12, 12,   ... % 4 symbols have length 12
    13, 13,           ... % 2 symbols have length 13
    14                ... % 1 symbol has length 14
];

% Verification: Check that the total count is 15
if length(codeword_lengths) ~= 15
    error('The number of symbols in the data vector must be exactly 15.');
end

% --- 2. Define Bins and Plot the Histogram ---
min_length = min(codeword_lengths);
max_length = max(codeword_lengths);

% Define bin edges to center bars on integer lengths (e.g., bin 10 runs from 9.5 to 10.5)
bin_edges = (min_length - 0.5) : 1 : (max_length + 0.5); 

figure; % Create a new figure window

% Plot the histogram
h = histogram(codeword_lengths, 'BinEdges', bin_edges, ...
              'FaceColor', [0.1 0.5 0.7], ... % Set bar color
              'EdgeColor', 'k');             % Black outline

% --- 3. Customize the Plot ---
xlabel('Codeword length (bits)', 'FontSize', 12);
ylabel('Number of symbols (Total = 15)', 'FontSize', 12);
title('Codeword Length Distribution (N=15)', 'FontSize', 14);

% Set X-axis ticks to be integers
xticks(min_length:1:max_length);

% Adjust Y-axis limit for clarity (max frequency is 5)
ylim([0 6]); 
grid on;
