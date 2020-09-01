folder = 'C:\Users\\Devesh\Documents\PatternRecognition\project2\speech\test';
wav_concat = '.wav';
feature = [];
cepFeatures = cepstralFeatureExtractor;
    

train_features = [];
test_features = [];
features = [];

vec1 = randperm(3600,720);
vec1 = vec1 - 1;
vec2 = randperm(3600,720);
vec2 = vec2 + 3599;
vec = [vec1, vec2];
vec = sort(vec);
test_samples = [];


for sample = 1 : 6000
    
    file = strcat(folder, int2str(sample), wav_concat);
    [x, Fs] = audioread(file);
    
    %% Time Domain Features
   
    var = 2;
    j = -4:4;
    p = -j.^2/(2*var);
    filt = (1/sqrt(2*pi*var)).*(exp(p));
    N = size(x, 1);
    x = [zeros(1, 4), x', zeros(1, 4)];
    i = 1;
    denoised = zeros(N, 1);
    while i <= N
        denoised(i) = sum(x(i:i+8).*filt);
        i = i + 1;
    end
    
    denoised = denoised';
    avgSignal = [];
    windowSize = 161;
    slide = 80;
    i = 1;
    var = 200;
    j = -80:80;
    p = -j.^2/(2*var);
    filt = (1/sqrt(2*pi*var)).*(exp(p));
    while i <= size(denoised, 2) - windowSize
        avgSignal = [avgSignal, sum(denoised(i:i+windowSize-1).*filt)];
        i = i + slide;
    end
    avg = mean(avgSignal);
    timeMax = max(avgSignal);
    timeMin = min(avgSignal);
    [~,peaklocs] = findpeaks(avgSignal);
    periodicity = mean(diff(avgSignal));
    ZCR = mean(abs(diff(sign(avgSignal))));
    
    %% Frequency Domain Features
    
    %FUNDAMENTAL FREQUENCY

    win_size = 161;
    slide_inc = 80;
    mean_win = [];
    num_win = ceil((length(denoised) - win_size)/slide_inc);
    for i = 1:num_win
        j = 1 + (i-1)*slide_inc;
        mean_win = [mean_win, mean(pitch(denoised', Fs))];
    end
    fund_freq = mean(mean_win);
    
    yfft = fftshift(denoised);

    %MAXIMUM FREQUENCY

    [max_yfft, max_index] = max(yfft);
    maximum = (max_index * Fs) / length(yfft);

    %MINIMUM FREQUENCY

    [min_yfft, min_index] = min(yfft);
    minimum = (min_index * Fs) / length(yfft);

    pow = yfft .* conj(yfft);
    sum_pow = sum(pow);

    [stft, normf, ~] = spectrogram(denoised,100,50,100,Fs);
    abs_stft = abs(stft);
    abs_normf = abs(normf);

    %FREQUENCY AVERAGE
    k = size(abs_stft,1);
    asa = zeros([k 1]);
    for i = 1:size(asa)
        asa(i) = mean(abs_stft(i,:));
    end
    favg = (abs_normf'*asa)/k;
 
    %FREQUENCY DEVIATION
 
    efx2 = ((abs_normf.*abs_normf)'*(asa.*asa))/k;
    fvar = efx2 - favg^2;
    fsd = sqrt(fvar);

    %% Cepstral Domain features
    
    % Pitch Estimation
    pit = [];
    win = hamming(length(denoised));
    windowedSignal = denoised.*win;
    fftResult=log(abs(fft(windowedSignal)));
    ceps=real(ifft(fftResult));
    nceps=length(ceps);
    
    peaks = zeros(nceps,1);
    k=3;
    
    while(k <= nceps/2 - 1)
        y1 = ceps(k - 1);
        y2 = ceps(k);
        y3 = ceps(k + 1);
        if (y2 > y1 && y2 >= y3)
            peaks(k)=ceps(k);
        end
    k=k+1;
    end
    [maxivalue, maxi]=max(peaks);
    pit = [pit; Fs/(maxi+1)];
    
    % formantFrequencies

    x1 = denoised'.*hamming(length(denoised));

    preemph = [1 0.63];
    x1 = filter(1,preemph,x1);

    A = lpc(x1,8);
    if sum(isnan(A)) == 0
        rts = roots(A);

        rts = rts(imag(rts)>=0);
        angz = atan2(imag(rts),real(rts));

        [frqs,indices] = sort(angz.*(Fs/(2*pi)));
        bw = -1/2*(Fs/(2*pi))*log(abs(rts(indices)));

        nn = 1;
        for kk = 1:length(frqs)
            if (frqs(kk) >= 0 && bw(kk) <= 8000)
                formants(nn) = frqs(kk);
                nn = nn+1;
            end
        end
    else
        formants = [0, 0, 0];
    end
    % MFCC
    
    [mfc, ~, ~] = cepFeatures(denoised');

    
    features = [avg, timeMax, timeMin, periodicity, ZCR, fund_freq, maximum, minimum, sum_pow, fsd, pit, formants(1:3), mfc'];
    if size(vec) ~=0 & vec(1) == sample
        test_features = [test_features; features];
        vec(1) = [];
    else
    train_features = [train_features; features];
    end
    disp(sample);
end
    
    