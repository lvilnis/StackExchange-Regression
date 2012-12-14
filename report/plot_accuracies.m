function[] = plot_accuracies()

close all;

if ~exist(fullfile(pwd, 'results.csv'))
	error('Missing results.csv file!');
else
	results = csvread(fullfile(pwd, 'results.csv'));
end
assert(size(results, 2) == 4);
dataset_column = 1;
classifier_column = 2;
partition_column = 3;
accuracy_column = 4;

nDatasets = unique(results(:, dataset_column));
nClassifiers = unique(results(:, classifier_column));
nPartitions = unique(results(:, partition_column));
color_str = {'bo-', 'ro-', 'go-', 'ko-', 'mo-', 'bs--', 'rs--', 'gs--', 'ks--', 'ms--'};
assert(length(nClassifiers) <= numel(color_str));
title_str = {'Dataset I', 'Dataset II', 'Dataset III'};

figure(); set(gcf, 'Position', [10, 100, 1200, 800]);
for d = 1:length(nDatasets)
	target_dataset = results(:, dataset_column) == nDatasets(d);
	for c = 1:length(nClassifiers)
		target_classifier = results(:, classifier_column) == nClassifiers(c);
		subplot(3, 1, d); plot(1:length(nPartitions),...
				results(target_dataset & target_classifier, accuracy_column),...
				color_str{c}, 'LineWidth', 2);
		grid on;
		if c == 1, hold on; ylim([50, 100]); end
		x_tick_str = {results(target_dataset & target_classifier, partition_column)'};
		set(gca, 'XTickLabel', x_tick_str);
	end
	if d == 1
		title(sprintf('Predicting closed questions\n%s', title_str{d}),...
					'FontSize', 13, 'FontWeight', 'b', 'FontName', 'Helvetica');
		h_legend = legend('svm(2)', 'Naive Bayes', 'Location', 'NorthWest', 'Orientation', 'Horizontal');
		set(h_legend, 'FontSize', 9);
	else
		title(sprintf('%s', title_str{d}),...
					'FontSize', 13, 'FontWeight', 'b', 'FontName', 'Helvetica');
	end
	if d == 2
		ylabel('Accuracies', 'FontSize', 13, 'FontWeight', 'b', 'FontName', 'Helvetica');
	end
	if d == length(nDatasets)
		xlabel('Training set partition', 'FontSize', 13, 'FontWeight', 'b', 'FontName', 'Helvetica');
	end
end

file_name = sprintf('images/stackoverflow_results');
% print(gcf, '-dpdf', '-painters', file_name);
image_format = 'png';
savesamesize(gcf, 'file', file_name, 'format', sprintf('-d%s', image_format));

