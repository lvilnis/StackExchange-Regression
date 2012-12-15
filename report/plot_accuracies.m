function[] = plot_accuracies()

close all;

if ~exist(fullfile(pwd, 'results.csv'))
	error('Missing results.csv file!');
else
	results = csvread(fullfile(pwd, 'results.csv'));
end
assert(size(results, 2) == 4);

classifier_column = 1;
partition_column = 2;
accuracy_column = 3;
dataset_column = 4;
results = sortrows(results, partition_column)
nDatasets = unique(results(:, dataset_column));
nClassifiers = unique(results(:, classifier_column));
nPartitions = unique(results(:, partition_column));
color_str = {'bo-', 'ro-', 'go-', 'ko-', 'mo-', 'bs--', 'rs--', 'gs--', 'ks--', 'ms--'};
assert(length(nClassifiers) <= numel(color_str));
title_str = {'Prediction For Closed Questions', 'Prediction For Open Questions', 'Dataset III'};

figure(); set(gcf, 'Position', [10, 100, 1200, 800]);
for d = 1:2
	target_dataset = results(:, dataset_column) == nDatasets(d);
	for c = 1:length(nClassifiers)
		target_classifier = results(:, classifier_column) == nClassifiers(c);
		subplot(2, 1, d); plot(results(target_dataset & target_classifier, partition_column),...
				results(target_dataset & target_classifier, accuracy_column),...
				color_str{c}, 'LineWidth', 2);
		grid on;
		if c == 1, hold on; ylim([.6, .85]); end
        %x_tick_str = { linspace(1,10)};
		%x_tick_str = {results(target_dataset & target_classifier, partition_column)'};
		%set(gca, 'XTickLabel', x_tick_str);
    end
    if d == 1
        title(sprintf('F1 Scores With Increasing Data Set Size\n\n\n%s', title_str{d}),...
					'FontSize', 13, 'FontWeight', 'b', 'FontName', 'Helvetica');
    end
    if d == 2
            title(sprintf( title_str{d}),...
					'FontSize', 13, 'FontWeight', 'b', 'FontName', 'Helvetica');
    end
    h_legend = legend('Naive Bayes', 'Log Loss w/ AdaGrad', 'Hinge Loss w/ AdaGrad','Liblinear SVM', 'Location', 'NorthWest', 'Orientation', 'Horizontal');
	set(h_legend, 'FontSize', 9);
	
	ylabel('F1', 'FontSize', 13, 'FontWeight', 'b', 'FontName', 'Helvetica');
	
    if d == 2
        xlabel(sprintf('\n\n70/30 Train/Test Split'), 'FontSize', 13, 'FontWeight', 'b', 'FontName', 'Helvetica');
    end
end

file_name = sprintf('images/stackoverflow_results');
% print(gcf, '-dpdf', '-painters', file_name);
image_format = 'png';
savesamesize(gcf, 'file', file_name, 'format', sprintf('-d%s', image_format));

