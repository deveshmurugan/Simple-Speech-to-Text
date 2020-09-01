dpm = loadmat('testAcc2.mat');
mdl = dpm.Mdl2;

x = audioread(file);


start = 1;
last = 8000;
stop = 0;

while(stop ~= 1)
    
    xi = x(start:last);
    features = fe(xi);
    x3 = transform(mdl, features);
    output = [];    
    a = sqrt(sum((x3 - m1).^2));
    b = sqrt(sum((x3 - m2).^2));
    if a <= b
        output = [output, 'bus', ', '];
    else
        output = [output, 'train', ', '];
    end

    disp(output);


    start = start + 6000;
    last = last + 8000;
        if last > length(x)
            last = length(x); 
            stop = 1;
        end
end



