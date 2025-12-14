function [odg, movb] = PEAQTest(ref, test)
% PEAQTest  - uruchamia pomiar PEAQ dla dwóch plików audio
% Użycie: [odg, movb] = PEAQTest('ref.wav', 'test.wav')

    [odg, movb] = PQevalAudio_fn(ref, test);

end