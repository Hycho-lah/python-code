hist(p_values);
title('Histogram of p-values');
xlabel('p-values') % x-axis label;
ylabel('Frequency');

count = 0;
for g = 1:length(p_values)
    if p_values(g) <= 0.05
        count = count + 1;
    end
end

FDR_pvalues = mafdr(p_values,'BHFDR', true);

FDR_count = 0;
for g = 1:length(FDR_pvalues)
    if FDR_pvalues(g) <= 0.05
        FDR_count = FDR_count + 1;
    end
end

gene_names_string = [""];
for i = 1:length(gene_names)
    gene_names_string(end+1) = string(gene_names{i});
end
gene_names_string = gene_names_string(2:end);
gene_names_string = transpose(gene_names_string);

fdr_value_genes = [FDR_pvalues,gene_names_string];
fdr_value_genes_sorted = sortrows(fdr_value_genes);

genes_top_diff = fdr_value_genes_sorted(1:10,2);
pos_top_diff_genes = [];
for g = 1:length(genes_top_diff)
    for p=1:length(gene_names)
        if genes_top_diff(g) == gene_names(p)
            pos_top_diff_genes(end+1) = p;
        end
    end
end
mean2fold_top = [];
log2foldchanges_top = [];
for i = 1: length(pos_top_diff_genes)
    p = pos_top_diff_genes(i);
    log2foldchanges_top(end+1) = log2(mean_array_cancer(p)/mean_array_normal(p));
    mean2fold_top(end+1) = log2(mean(counts(i,:)));
end

figure; 
scatter(mean_count_up_log2,log2foldchange_up_array);
hold on;
scatter(mean_count_down_log2,log2foldchange_down_array);
scatter(mean_count_others_log2,log2foldchange_others_array);
scatter(mean2fold_top,log2foldchanges_top,'k');
xlabel('Log2 Mean');
ylabel('Log2 Fold Change');
title('Log2 Mean vs. Log2 Fold Change');

