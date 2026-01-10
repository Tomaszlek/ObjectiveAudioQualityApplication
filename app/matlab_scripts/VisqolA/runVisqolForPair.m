function mos = runVisqolForPair(refFile, degFile)
    [refSignal, fs1] = audioread(refFile);
    [degSignal, fs2] = audioread(degFile);

    mos = visqol(refSignal, degSignal, fs1);
end
