data = xlsread('training.xls');

Mdl = sparsefilt(data,10);
x = transform(Mdl, data);
[dpmm, dpmmp, t] = DPMM_gauss(x);


labels = dpmm.nk;
meanVector1 = mean(x(1:labels(1), :));
meanVector2 = mean(x(labels(1):end, :));

test = xlsread('testing.xls');
Mdl2 = sparsefilt(test,10);
x2 = transform(Mdl, test);
testLabels = [];
for i=1:100
    a = sqrt(sum((x2(i, :) - meanVector1).^2));
    b = sqrt(sum((x2(i, :) - meanVector2).^2));
    if a <= b
        testLabels = [testLabels, 1];
    else
        testLabels = [testLabels, 2];
    end
end
a = 0;
truth = [ones(1, 50), 2*ones(1, 50)];
for i=1:100
    if truth(i) == testLabels(i)
        a = a+1;
    end
end
