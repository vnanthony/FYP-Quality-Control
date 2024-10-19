%Purposes: Data Filtering for Baseline Drift Correction and Noise Minimization, and Feature Extraction
marked=[];
k=0;
area1=0;
area2=0;
for i=1:30000
    ppg=ppgcohort0processed(i,:);
    %Baseline Correction and Noise Reduction
    % Baseline Correction and Noise Reduction
    % 1. Chebyshev filter design
    Fs = 1 / 0.1;  % Sampling rate is the inverse of the mean time difference
    Fh = 0.5;  % High-pass cutoff
    Fl = 5;    % Low-pass cutoff
    Fn = Fs / 2;   % Nyquist frequency
    [A, B, C, D] = cheby2(6, 20, [Fh Fl] / Fn);  % Chebyshev type II filter
    [filter_sos, g] = ss2sos(A, B, C, D);        % Convert to second-order sections
    
    % Apply filter to the PPG signal
    newppg1 = filtfilt(filter_sos, g, ppg);
    figure(4);
    plot(newppg1);
    
    % 2. Moving average filter for smoothing
    windowWidth = 8;  % Moving average window
    kernel = ones(windowWidth, 1) / windowWidth;
    newppg2 = filter(kernel, 1, newppg1);
    figure(5);
    plot(newppg2);

    % Finding peaks of the PPG, VPG, and APG
    [pks1, locs1] = findpeaks(newppg2, Fs);  % PPG peaks

    % First derivative (VPG)
    dnewppg2 = diff(newppg2);
    [pks2, locs2] = findpeaks(dnewppg2, Fs);  % VPG peaks

    % Second derivative (APG)
    dnewppg3 = diff(newppg2, 2);
    [pks3, locs3] = findpeaks(dnewppg3, Fs);  % APG peaks

    % Quality assessment and feature extraction
    if numel(locs2) >= (2 * numel(locs1) - 8) && numel(locs2) <= (2 * numel(locs1) + 2)
        % Heart rate calculation
        k = k + 1;
        [pks, locs] = findpeaks(newppg2, Fs, 'MinPeakDistance', 0.273 * Fs);
        
        % Feature 1: Heart rate estimation
        feature(k, 1) = (numel(locs2) / 2) / (10 / 60);  % Beats per minute
        
        % Feature 2: Mean amplitude of systolic peaks
        feature(k, 2) = mean(pks);
        
        % Systolic-diastolic amplitude and time interval between systolic and diastolic peaks
        difference = [];
        time_interval = [];
        a = numel(locs2);
        b = 1;
        while b < a
            if (pks2(b) - pks2(b + 1)) > 0
                for m = 1:numel(locs)
                    if locs(m) > locs2(b) && locs(m) < locs2(b + 1)
                        difference = [difference, pks(m) - newppg2(round(locs2(b + 1) * Fs))];
                        time_interval = [time_interval, locs2(b + 1) - locs(m)];
                    end
                end
                b = b + 2;
            elseif b < (a - 1) && (pks2(b + 1) - pks2(b + 2)) > 0
                for m = 1:numel(locs)
                    if locs(m) > locs2(b + 1) && locs(m) < locs2(b + 2)
                        difference = [difference, pks(m) - newppg2(round(locs2(b + 2) * Fs))];
                        time_interval = [time_interval, locs2(b + 2) - locs(m)];
                    end
                end
                b = b + 3;
            else
                b = b + 2;
            end
        end
        
        % Feature 3: Mean systolic-diastolic difference
        mean_difference = mean(difference);
        feature(k, 3) = mean_difference;

        % Feature 4: Mean time interval between systolic and diastolic
        mean_time_interval = mean(time_interval);
        feature(k, 4) = mean_time_interval;

        % Feature 5: Mean amplitude between diastolic peaks and onsets
        feature(k, 5) = feature(k, 2) - feature(k, 3);

        % Feature 6: Ratio (systolic-diastolic) / systolic
        feature(k, 6) = feature(k, 3) / feature(k, 2);

        %%
        %AUC 
        minus_newppg2 = -newppg2;
        [pks_minus,locs_minus]=findpeaks(minus_newppg2,125,'minpeakdistance',0.273); 
        %area under curve 
        try
            if pks2(1) < pks2(2)
                x1 = round(locs_minus(1) * Fs);
                x2 = round(locs2(3) * Fs);
                area1 = trapz(x1:x2, newppg2(x1:x2));

                x1 = round(locs2(3) * Fs);
                x2 = round(locs_minus(2) * Fs);
                area2 = trapz(x1:x2, newppg2(x1:x2));
            elseif pks2(1) > pks2(2) && locs_minus(1) < locs(1)
                x1 = round(locs_minus(1) * Fs);
                x2 = round(locs2(2) * Fs);
                area1 = trapz(x1:x2, newppg2(x1:x2));

                x1 = round(locs2(2) * Fs);
                x2 = round(locs_minus(2) * Fs);
                area2 = trapz(x1:x2, newppg2(x1:x2));
            end
        catch
            continue;
        end

        % Feature 7: Total area under the curve
        feature(k, 7) = area1 + area2;

        % Mark successful data
        marked = [marked, i];
    end
end

marked=marked';

