function mos = runVisqolForPair(refFile, degFile)
    % --- Wczytanie sygnałów ---
    [refSignal, fs1] = audioread(refFile);
    [degSignal, fs2] = audioread(degFile);

    % --- Resampling jeśli różne sr ---
    if fs1 ~= fs2
        [P,Q] = rat(fs1/fs2);
        degSignal = resample(degSignal,P,Q);
    end

    % --- Dopasowanie długości ---
    minLen = min(length(refSignal), length(degSignal));
    refSignal = refSignal(1:minLen,:);
    degSignal = degSignal(1:minLen,:);

    % --- Konwersja do mono ---
    if size(refSignal,2) > 1, refSignal = mean(refSignal,2); end
    if size(degSignal,2) > 1, degSignal = mean(degSignal,2); end

    % --- Normalizacja ---
    refSignal = refSignal / max(abs(refSignal), [], 'all');
    degSignal = degSignal / max(abs(degSignal), [], 'all');

    % --- VISQOL ---
    mos = visqol(refSignal, degSignal, fs1);
end
