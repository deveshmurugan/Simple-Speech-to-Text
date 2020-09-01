
folder = 'C:\Users\Devesh\Documents\PatternRecognition\project2\testChunks\speech';
wav_concat = '.wav';
sample = 2;
file = strcat(folder, int2str(sample), wav_concat);
[x, Fs] = audioread(file);

labelArray = [];
distanceArray = [];

start = 1;
last = 8000;
stop = 0;

while(stop ~= 1)
    
    disp(start);
    disp(last);
    xi = x(start:last);
    [label, distance] = predictPoint(xi);
    labelArray = [labelArray, label];
    distanceArray = [distanceArray, distance];
    start = start + 6000;
    last = last + 6000;
        if last > length(x)
            last = length(x); 
            stop = 1;
        end
end

mask = [-1, 1];
edge = [];
for i=1:length(labelArray) - 1
    temp = labelArray(i:i+1);
    edge = [edge, abs(sum(temp.*mask))];
end
word = edge;
word(word == 0) = 2;
x = labelArray; 
i = 1;
time = [];
while i < length(x)
    sub = [];
    if x(i) == x(i+1)
        c = 2;
        j = i;
        sub = [sub, j];
        while (x(j) == x(j+1))
            sub = [sub, j+1];
            j = j + 1;
        end
        y = mink(distanceArray(sub(1):sub(end)), c);
        for i=1:c
            minid = find(distanceArray == y(i));
            time = [time, minid];
        end
        i = sub(end) + 1;
        else
        time = [time, i];
        i = i + 1;
    end
end
time = [time, i];

