counts = importdata("size_factor_counts.txt");
labels = importdata("labels.txt");
gene_names = importdata("GeneNames.txt");
[m,n] = size(counts);

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

tLocal = nbintest(cancer_counts,normal_counts,'VarianceLink','LocalRegression');
plotVarianceLink(tPoisson,'Compare',true)
p_values = tLocal.pValue;

