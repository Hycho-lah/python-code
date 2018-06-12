counts = importdata("size_factor_counts.txt");
labels = importdata("labels.txt");
gene_names = importdata("GeneNames.txt");

[m,n] = size(counts);
% create separate matrices based on 1) cancer 2) normal 
cancer_counts = double.empty(0,0);
normal_counts = double.empty(0,0);
%normal_counts =[]

for i = 1:length(labels)
    if labels(i) == 1
        cancer_counts = cat(2,cancer_counts,counts(:,i));
    elseif labels(i) == 2 
        normal_counts = cat(2,normal_counts,counts(:,i));
    end
end
mean_array_cancer = [];
disp_array_cancer = [];
for j = 1:length(cancer_counts)
   gene = cancer_counts(j,:);
   mean_gene = mean(gene);
   std_gene = std(gene);
   disp_gene = std_gene/mean_gene;
   mean_array_cancer(end+1) = mean_gene;
   disp_array_cancer(end+1) = disp_gene;
end
mean_array_cancer_log2 = log2(mean_array_cancer);
disp_array_cancer_log2 = log2(disp_array_cancer);

mean_array_normal = [];
disp_array_normal = [];
for j = 1:length(normal_counts)
   gene = normal_counts(j,:);
   mean_gene = mean(gene);
   std_gene = std(gene);
   disp_gene = std_gene/mean_gene;
   mean_array_normal(end+1) = mean_gene;
   disp_array_normal(end+1) = disp_gene;
end
mean_array_normal_log2 = log2(mean_array_normal);
disp_array_normal_log2 = log2(disp_array_normal);

figure; 
scatter(disp_array_cancer_log2,mean_array_cancer_log2);
hold on;
scatter(disp_array_normal_log2,mean_array_normal_log2);
xlabel('Dispersion log2');
ylabel('Mean log2');

% foldchange = cancer_mean_of_gene_i / normal_mean_of_gene_i
log2foldchange_up_array = [];
mean_count_up_log2 =[];
log2foldchange_down_array = [];
mean_count_down_log2 =[];
gene_name_up = [""];
gene_name_down = [""];
log2foldchange_others_array = [];
mean_count_others_log2 =[];

for i = 1:length(mean_array_normal_log2)
    log2foldchange = log2(mean_array_cancer(i)/mean_array_normal(i));
    if log2foldchange > 1
        log2foldchange_up_array(end+1) = log2foldchange;
        mean_count_up_log2(end+1) = log2(mean(counts(i,:)));
        gene_name = string(gene_names{i});
        gene_name_up(end+1) = gene_name;
    elseif log2foldchange < -1
        log2foldchange_down_array(end+1) = log2foldchange;
        mean_count_down_log2(end+1) = log2(mean(counts(i,:)));
        gene_name = string(gene_names{i});
        gene_name_down(end+1) = gene_name;
    else 
        log2foldchange_others_array(end+1) = log2foldchange;
        mean_count_others_log2(end+1) = log2(mean(counts(i,:)));
    end
end

gene_name_up = gene_name_up(2:end);
gene_name_down = gene_name_down(2:end);

genefold_up= transpose([log2foldchange_up_array;gene_name_up]);
genefold_down= transpose([log2foldchange_down_array;gene_name_down]);

genefold_up_sorted = sortrows(genefold_up);
genefold_down_sorted = sortrows(genefold_down);

figure; 
scatter(mean_count_up_log2,log2foldchange_up_array);
hold on;
scatter(mean_count_down_log2,log2foldchange_down_array);
scatter(mean_count_others_log2,log2foldchange_others_array,'k');
xlabel('Log2 Mean');
ylabel('Log2 Fold Change');
title('Log2 Mean vs. Log2 Fold Change');


