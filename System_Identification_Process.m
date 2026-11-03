%% PQM-1B system identification
clearvars; close all; clc;

%% Configuration
pqmRoot = "/Users/williamyelland/Thesis/Modelling/PQM-1B/PQM-1B";
simRoot = "/Users/williamyelland/Thesis/Modelling/Simulation Model";

stepFolders = ["Freq Step 1Hz","Freq Step 0.005Hz","Freq Step 0.015Hz", ...
    "Freq Step 0.025Hz","Freq Step 0.035Hz","Freq Step 0.045Hz", ...
    "Freq Step -1Hz","Freq Step -0.005Hz","Freq Step -0.015Hz", ...
    "Freq Step -0.025Hz","Freq Step -0.035Hz","Freq Step -0.045Hz"];
stepLabels  = ["+1","+0.005","+0.015","+0.025","+0.035","+0.045", ...
    "-1","-0.005","-0.015","-0.025","-0.035","-0.045"];
rampFolder  = "Freq Ramp 4Hz per s";
rampTargets = ["48 Hz","49 Hz","49.5 Hz","49.98 Hz","50.02 Hz","50.5 Hz","51 Hz","52 Hz"];

trainSign = 1;                % +1 trains on over-frequency, validates on under

simInputFile  = "test1.csv";
simInputAlt   = "Test 1.csv";
simValueName  = 'Freg';       % the actual simulated frequency, not the setpoint
pqmOutputName = 'Freq_PQM1B';

maxRepeats = 20;
tMin = 0.0;  tMax = 3.5;  TsTarget = 1e-3;   % decimate by selection, no filter

nkFallback = 52;              % samples, where a delay measurement fails outright
delayBaselineWindow = 0.4;    % s averaged to find each record's operating point
delayWindow = [0.025 0.150];  % s, the lag a record is physically allowed to show
gainSettleSec = 0.25;         % input must hold still this long to count toward gain
useDataGain = true;           % rescale the winner's DC gain to the measured one

runLinear = true;  runHW = true;  runNLARX = true;
tfestPoleGrid = 1:15;                        % zeros run 0:(nPoles-1)
arxNaRange = 1:15;  arxNbRange = 0:15;
hwNbGrid   = 1:15;  hwNfGrid   = 1:15;
hwNlNames  = {'sigmoidnet','wavenet','pwlinear','poly1d'};
hwNlCtors  = {@() idSigmoidNetwork, @() idWaveletNetwork, @() idPiecewiseLinear, @() idPolynomial1D};
nlarxNames = {'TreePartition','sigmoidnet','wavenet','linear'};
nlarxCtors = {@() idTreePartition, @() idSigmoidNetwork, @() idWaveletNetwork, @() idLinear};

keepFit   = 85;               % drop a candidate below this on train or val
targetFit = 85;               % stop sweeping at the first candidate clearing both
delayTargetFit = targetFit;   % the bar with the lag back on, which decides what ships
maxDelayTrials = 12;
sweepIter  = 20;  sweepTol  = 1e-5;  tfestSweepTol  = 1e-2;
refineIter = 50;  refineTol = 1e-8;  tfestRefineTol = 1e-5;

numWorkers = 5;               % the pool reads CSVs only; estimation stays serial
cacheFile = 'data_cache.mat';
delayFile = 'delay_estimate.mat';
saveFigures = true;

rng(1);
warning('off', 'Ident:estimation:MultiExpIntersampleDifference');
if isempty(gcp('nocreate'))
    try
        parpool('local', numWorkers);
    catch
        fprintf('No parallel pool - loading serially.\n');
    end
end

%% Load
datasets  = buildDatasets(pqmRoot, simRoot, stepFolders, stepLabels, rampFolder, ...
    rampTargets, trainSign, maxRepeats, simInputFile, simInputAlt);
stepNames = {datasets.plotLabel};
isTrain   = strcmp({datasets.role}, 'train');

% Anything in here that changes makes the cache stale.
boundary = struct('tMin', tMin, 'tMax', tMax, 'TsTarget', TsTarget, ...
    'simValueName', simValueName, 'pqmOutputName', pqmOutputName, 'decimate', 'select', ...
    'sets', {arrayfun(@(d) sprintf('%s|%s|%s|%d', d.label, d.role, d.pqmDir, d.numTests), ...
        datasets, 'UniformOutput', false)});

stepRaw = loadAll(datasets, boundary, cacheFile, tMin, tMax, TsTarget, ...
    pqmOutputName, simValueName);
dTrainRaw = merge(stepRaw{isTrain});
dValRaw   = merge(stepRaw{~isTrain});
Ts = sampleTime(dTrainRaw);
fprintf('%d train / %d val experiments from %d excitations at Ts = %g s.\n', ...
    numExp(dTrainRaw), numExp(dValRaw), numel(stepNames), Ts);

%% Delay
% nkPerTest is the lag per test, in samples. Models are fitted with nk = 0.
[nkPerSet, nkPerTest] = measureDelay(stepRaw, Ts, nkFallback, ...
    delayBaselineWindow, delayWindow);
nkAll   = [nkPerTest{:}];
nkModel = round(median(nkAll));
fprintf('Delay %d samples (%g ms), median of %d tests, per test %d to %d.\n', ...
    nkModel, nkModel*Ts*1e3, numel(nkAll), min(nkAll), max(nkAll));
save(delayFile, 'nkPerSet', 'nkPerTest', 'nkModel', 'Ts', 'stepNames');

%% Static gain in the data
% Only a transfer function can be split into K * Gn(s), so with runLinear off none
% of this has anywhere to go.
dataGain = NaN;  affineLaw = struct('gain', NaN, 'offset', NaN);
if runLinear
    [dataGain, affineLaw] = staticGain(stepRaw, Ts, gainSettleSec);
end
dataGainDev = dataGain - 1;

%% De-delay, then look at the data
stepFit = cellfun(@(d, n) removeDelay(d, n), stepRaw(:), nkPerTest(:), ...
    'UniformOutput', false);
dTrain = merge(stepFit{isTrain});
dVal   = merge(stepFit{~isTrain});

plotSets(stepRaw, stepNames, isTrain, [], stepRaw, 'Loaded data', '', ...
    pngName(saveFigures, 'data_overview_as_loaded'));
plotSets(stepFit, stepNames, isTrain, [], stepFit, 'Delay removed', '', ...
    pngName(saveFigures, 'data_overview_as_fitted'));

%% Candidates
candidates = buildCandidates(runLinear, runHW, runNLARX, tfestPoleGrid, ...
    arxNaRange, arxNbRange, hwNbGrid, hwNfGrid, hwNlNames, hwNlCtors, ...
    nlarxNames, nlarxCtors);
fprintf('%d candidates: %d LINEAR, %d HW, %d NLARX.\n', numel(candidates), ...
    sum(strcmp({candidates.type},'LINEAR')), sum(strcmp({candidates.type},'HW')), ...
    sum(strcmp({candidates.type},'NLARX')));

%% Sweep
optSweep  = estOptions(sweepIter,  sweepTol,  tfestSweepTol);
optRefine = estOptions(refineIter, refineTol, tfestRefineTol);
results = runSearch(candidates, dTrain, dVal, optSweep, keepFit, targetFit, Ts);
if isempty(results)
    error('SysID:noFit', 'No candidate cleared %g%% on both train and val.', keepFit);
end

%% Acceptance: the delay has to hold up too
[~, sortIdx] = sort([results.trainFit], 'descend');
results = results(sortIdx);
best = [];  bestScore = [];  runnerUp = [];
fprintf('\nAcceptance (%g%% on val with the delay back on):\n', delayTargetFit);
for i = 1:min(numel(results), maxDelayTrials)
    cand  = refineCandidate(results(i), candidates, dTrain, dVal, optRefine, Ts);
    score = scoreWithDelay(cand.sys, stepRaw, nkModel, isTrain);
    fprintf('  [%d] %-6s [%-10s] %-22s val %6.2f%% -> %6.2f%% (all %6.2f%%)\n', ...
        i, cand.type, num2str(cand.order), cand.nl, cand.valFit, score.val, score.all);
    if isempty(runnerUp) || (isfinite(score.val) && score.val > runnerUp.score.val)
        runnerUp = struct('cand', cand, 'score', score);
    end
    if isfinite(score.val) && score.val >= delayTargetFit
        best = cand;  bestScore = score;  break
    end
end
if isempty(best)
    best = runnerUp.cand;  bestScore = runnerUp.score;
    fprintf('  bar not cleared; best available is %.2f%% on val.\n', bestScore.val);
end
sysBest = best.sys;

%% Take the DC gain from the data, not from the fit
% tfest sets the DC gain to whatever minimises error over the whole record,
% transients included. dataGain is measured in steady state.
fittedGain = NaN;
if useDataGain && isfinite(dataGain)
    [fittedDcGain, sysUnit] = unityDcForm(sysBest);
    if isfinite(fittedDcGain)
        fittedGain    = fittedDcGain;
        sysRescaled   = dataGain * sysUnit;
        scoreRescaled = scoreWithDelay(sysRescaled, stepRaw, nkModel, isTrain);
        fprintf('DC gain %.10f -> %.10f (%+.4f mHz at 50 Hz); val %.2f%% -> %.2f%%.\n', ...
            fittedGain, dataGain, (dataGain - fittedGain)*50e3, bestScore.val, scoreRescaled.val);
        sysBest       = sysRescaled;
        best.trainFit = fitOf(sysBest, dTrain);
        best.valFit   = fitOf(sysBest, dVal);
        bestScore     = scoreRescaled;
    end
end

%% Report
fprintf('\nBest model: %s [%s] %s\n', best.type, num2str(best.order), best.nl);
fprintf('  no delay      train %.2f%%   val %.2f%%\n', best.trainFit, best.valFit);
fprintf('  delay back on train %.2f%%   val %.2f%%   all %d steps %.2f%%\n', ...
    bestScore.train, bestScore.val, sum(bestScore.tests > 0), bestScore.all);
[worstFit, iWorst] = min(bestScore.perSet + 1e9*(bestScore.tests == 0));
fprintf('  worst excitation: %s at %.2f%%\n', stepNames{iWorst}, worstFit);

%% Save and plot
modelGain = NaN;  sysLinUnit = [];
if runLinear
    [modelGain, sysLinUnit] = unityDcForm(sysBest);
end
modelGainDev = modelGain - 1;   % format short prints both gains as "1.0000"

if strcmp(best.type, 'LINEAR')
    modelDesc = sprintf('LTF - Np = %d, Nz = %d', best.order(1), best.order(2));
else
    modelDesc = sprintf('%s [%s] %s', best.type, strjoin(string(best.order), ' '), best.nl);
end
tag = sprintf('%s_%s_FIT%.2f_%s', best.type, strjoin(string(best.order), '-'), ...
    bestScore.val, char(datetime('now','Format','yyyyMMdd_HHmmss')));

plotSets(stepFit, stepNames, isTrain, sysBest, stepFit, ...
    sprintf('%s - no delay', modelDesc), ...
    sprintf('train FIT %.2f%%   |   val FIT %.2f%%', best.trainFit, best.valFit), ...
    pngName(saveFigures, tag));
stepDelayed = cellfun(@(d) delayInput(d, nkModel), stepRaw(:), 'UniformOutput', false);
plotSets(stepDelayed, stepNames, isTrain, sysBest, stepRaw, ...
    sprintf('%s + %d smp delay', modelDesc, nkModel), ...
    sprintf('train FIT %.2f%%   |   val FIT %.2f%%', bestScore.train, bestScore.val), ...
    pngName(saveFigures, ['DelayCombined_' tag]));

meta = struct('type', best.type, 'order', best.order, 'nl', best.nl, 'Ts', Ts, ...
    'trainFit', best.trainFit, 'valFit', best.valFit, ...
    'delayNk', nkModel, 'delayNkSec', nkModel*Ts, 'delayPerSet', nkPerSet, ...
    'finalTrainFit', bestScore.train, 'finalValFit', bestScore.val, ...
    'finalAllFit', bestScore.all, 'finalPerSet', bestScore.perSet, ...
    'testsPerSet', bestScore.tests, 'modelGain', modelGain, ...
    'modelGainDev', modelGainDev, 'fittedGain', fittedGain, 'useDataGain', useDataGain, ...
    'dataGain', dataGain, 'dataGainDev', dataGainDev, ...
    'affineGain', affineLaw.gain, 'affineOffset', affineLaw.offset);
[sysDelay, sysFinal] = delayModel(sysBest, nkModel, Ts);
matPath  = sprintf('BestModel_%s.mat', tag);
saveVars = {'sysBest', 'sysDelay', 'sysFinal', 'nkModel', 'meta'};
if runLinear
    saveVars = [saveVars, {'sysLinUnit', 'modelGain', 'modelGainDev', 'dataGain', 'dataGainDev'}];
end
save(matPath, saveVars{:});
fprintf('Saved %s\n', matPath);

%% The winner as K * Gn(s), for RSCAD
% The static bias is the ~5e-5 gap between the numerator and denominator constants,
% which four significant figures round away. Splitting K off keeps it a number you
% set on the block's gain field.
if isfinite(modelGain)
    [numUnit, denUnit] = tfdata(sysLinUnit, 'v');
    fprintf('\nG(s) = K * Gn(s)\n');
    fprintf('  K   = %.10f  (%+.4f mHz at 50 Hz)\n', modelGain, modelGainDev*50e3);
    fprintf('  num = %s\n  den = %s\n', mat2str(numUnit, 10), mat2str(denUnit, 10));
else
    fprintf('\n%s has no transfer function, so no K * Gn(s) form.\n', best.type);
end

%% Helper functions

function datasets = buildDatasets(pqmRoot, simRoot, stepFolders, stepLabels, ...
    rampFolder, rampTargets, trainSign, maxRepeats, simPref, simAlt)
    % One row per directory, assigned whole to train or val by its direction.
    names   = [stepLabels + " Hz", "ramp " + rampTargets];
    pqmDirs = [fullfile(pqmRoot, stepFolders), fullfile(pqmRoot, rampFolder, rampTargets)];
    simDirs = [fullfile(simRoot, stepFolders), fullfile(simRoot, rampFolder, rampTargets)];
    signs   = [sign(str2double(stepLabels)), sign(str2double(erase(rampTargets," Hz")) - 50)];

    datasets = struct('label',{}, 'plotLabel',{}, 'role',{}, 'pqmDir',{}, ...
        'simFile',{}, 'numTests',{});
    for i = 1:numel(names)
        pqmDir = char(pqmDirs(i));  nTests = 0;
        while nTests < maxRepeats && isfile(fullfile(pqmDir, sprintf('test%d.csv', nTests+1)))
            nTests = nTests + 1;
        end
        if nTests < 1, continue; end
        simFile = char(fullfile(simDirs(i), simPref));
        if ~isfile(simFile), simFile = char(fullfile(simDirs(i), simAlt)); end
        if ~isfile(simFile)
            error('SysID:noSimInput', 'No "%s" or "%s" in %s.', simPref, simAlt, simDirs(i));
        end
        role = 'val';   % an excitation with no direction has nothing to sort on
        if signs(i) ~= 0 && signs(i) == sign(trainSign), role = 'train'; end
        datasets(end+1,1) = struct('label', char(names(i)), ...
            'plotLabel', sprintf('%s [%s]', names(i), role), 'role', role, ...
            'pqmDir', pqmDir, 'simFile', simFile, 'numTests', nTests); %#ok<AGROW>
    end
    if ~any(strcmp({datasets.role},'train')), error('SysID:noTrain','No training excitation.'); end
    if ~any(strcmp({datasets.role},'val')),   error('SysID:noVal','No validation excitation.'); end
end

function stepRaw = loadAll(datasets, boundary, cacheFile, tMin, tMax, TsTarget, ...
    outName, simName)
    if isfile(cacheFile)
        cached = load(cacheFile);
        if isfield(cached,'boundary') && isequal(cached.boundary, boundary)
            stepRaw = cached.stepRaw;
            fprintf('Loaded data from %s.\n', cacheFile);
            return
        end
    end
    stepRaw = cell(numel(datasets), 1);
    for s = 1:numel(datasets)
        cells = loadDir(datasets(s), tMin, tMax, TsTarget, outName, simName);
        stepRaw{s} = merge(cells{:});
    end
    save(cacheFile, 'stepRaw', 'boundary', '-v7.3');
end

function cells = loadDir(ds, tMin, tMax, TsTarget, outName, simName)
    % Every repeat in one directory, windowed and decimated by selection.
    [simT, simU] = loadSimTrace(ds.simFile, simName);
    nTests = ds.numTests;  pqmDir = ds.pqmDir;
    cells = cell(nTests, 1);
    parfor i = 1:nTests
        raw = readtable(fullfile(pqmDir, sprintf('test%d.csv', i)), ...
            'VariableNamingRule', 'preserve');
        t = raw{:, 1};
        inWindow = t >= tMin & t <= tMax;
        tWin  = t(inWindow);
        TsRaw = mean(diff(tWin));
        decim = round(TsTarget / TsRaw);
        if decim < 1
            error('SysID:TsTooFine', 'TsTarget %g is finer than the raw %g.', TsTarget, TsRaw);
        end
        y = raw{inWindow, resolveCol(raw.Properties.VariableNames, outName)};
        u = interp1(simT, simU, tWin, 'linear');
        u(tWin < simT(1)) = simU(1);  u(tWin > simT(end)) = simU(end);
        cells{i} = iddata(y(1:decim:end), u(1:decim:end), decim*TsRaw, 'TimeUnit', 'seconds');
    end
end

function [simT, simU] = loadSimTrace(file, name)
    raw  = readtable(file, 'VariableNamingRule', 'preserve');
    simT = double(raw{:, 1});
    simU = double(raw{:, resolveCol(raw.Properties.VariableNames, name)});
    ok   = isfinite(simT) & isfinite(simU);
    simT = simT(ok);  simU = simU(ok);
    [simT, sortIdx] = sort(simT);              simU = simU(sortIdx);
    [simT, uniqIdx] = unique(simT, 'stable');  simU = simU(uniqIdx);
end

function col = resolveCol(colNames, wanted)
    % Columns export as '...|<name>'; endsWith, so 'Freg' cannot match 'Freq_PQM1B'.
    names = string(colNames);
    idx = find(endsWith(names, string(wanted)));
    if isempty(idx), idx = find(contains(names, string(wanted))); end
    if isempty(idx), error('SysID:noColumn', 'No column "%s".', wanted); end
    col = idx(1);
end

function [nkPerSet, nkPerTest] = measureDelay(stepRaw, Ts, fallback, baseWin, win)
    % delayest is the primary estimator, the response onset stands behind it, and
    % anything outside the physical window goes to the onset detector instead.
    nSets = numel(stepRaw);
    nkPerSet = zeros(1, nSets);  nkPerTest = cell(nSets, 1);
    nkLo = max(0, round(win(1)/Ts));  nkHi = round(win(2)/Ts);
    nTotal = 0;  nOnset = 0;  nDropped = 0;
    for s = 1:nSets
        nRep = numExp(stepRaw{s});
        lags = nan(1, nRep);
        for e = 1:nRep
            rec = getexp(stepRaw{s}, e);
            u = rec.u(:);  y = rec.y(:);
            % The operating point comes off here and nowhere else, on a throwaway
            % copy: the ~50 Hz pedestal would collapse delayest's answer to nk = 2.
            nBase = max(1, min(numel(y), round(baseWin/Ts)));
            dDetrended = iddata(y - mean(y(1:nBase)), u - mean(u(1:nBase)), Ts);
            nkFromDelayest = NaN;
            try
                nkFromDelayest = max(delayest(dDetrended, 2, 2, 0, max(nkHi, 40)));
            catch
            end
            nTotal = nTotal + 1;
            if isfinite(nkFromDelayest) && nkFromDelayest >= nkLo && nkFromDelayest < nkHi
                lags(e) = nkFromDelayest;
                continue
            end
            nkFromOnset = onsetDelay(u, y, nBase);
            if isfinite(nkFromOnset) && nkFromOnset >= nkLo && nkFromOnset < nkHi
                lags(e) = nkFromOnset;  nOnset = nOnset + 1;
            else
                nDropped = nDropped + 1;
            end
        end
        ok = isfinite(lags);
        if ~any(ok)
            nkPerSet(s) = fallback;  nkPerTest{s} = repmat(fallback, 1, nRep);
            continue
        end
        nkPerSet(s) = round(median(lags(ok)));
        lags(~ok) = nkPerSet(s);
        nkPerTest{s} = round(lags);
    end
end

function nk = onsetDelay(u, y, nBase)
    % The samples between the input leaving its baseline and the output leaving its
    % own. A step either moved the output or it did not.
    nk = NaN;
    uBase = mean(u(1:nBase));  yBase = mean(y(1:nBase));
    uStep = max(abs(u - uBase));
    if ~(uStep > 0), return; end
    iInput = find(abs(u - uBase) > 0.02*uStep, 1);
    if isempty(iInput), return; end
    % 2 mHz floor: two counts of the 1 mHz grid the output sits on, so quantisation
    % alone can never trip it.
    thr = max(4*std(y(1:nBase)), 0.002);
    iOutput = iInput - 1 + find(abs(y(iInput:end) - yBase) > thr, 1);
    if isempty(iOutput), return; end
    nk = iOutput - iInput;
end

function data = removeDelay(data, nShift)
    % Advances y against u so the edges line up; the record comes back shorter.
    nRep = numExp(data);
    if isscalar(nShift), nShift = repmat(nShift, 1, nRep); end
    if all(nShift <= 0), return; end
    parts = cell(1, nRep);
    for e = 1:nRep
        rec = getexp(data, e);  shift = nShift(e);
        if shift <= 0, parts{e} = rec; continue; end
        y = rec.y;  nSamp = size(y, 1);
        trimmed = rec(1:nSamp-shift);
        trimmed.OutputData = y(shift+1:end, :);
        parts{e} = trimmed;
    end
    data = merge(parts{:});
end

function data = delayInput(data, nShift)
    % Pushes u later, the inverse of removeDelay. The samples sliding in at the
    % front are held at u(1), so nothing is trimmed.
    if nShift <= 0, return; end
    nRep = numExp(data);  parts = cell(1, nRep);
    for e = 1:nRep
        rec = getexp(data, e);  u = rec.u;
        shifted = rec;
        shifted.InputData = [repmat(u(1,:), nShift, 1); u(1:end-nShift, :)];
        parts{e} = shifted;
    end
    data = merge(parts{:});
end

function [gainAvg, affine] = staticGain(stepRaw, Ts, settleSec)
    % The gain the data shows, from windows where the input has been still longer
    % than the transport lag, so no alignment is needed. A ratio through the origin
    % cannot carry an offset, so the affine law comes with it.
    nSettle = max(2, round(settleSec/Ts));
    gainPerSet = [];  inputAll = [];  outputAll = [];
    for s = 1:numel(stepRaw)
        gains = [];
        for e = 1:numExp(stepRaw{s})
            rec = getexp(stepRaw{s}, e);  u = rec.u(:);  y = rec.y(:);
            mask = settledMask(u, nSettle);
            if ~any(mask), continue; end
            gains(end+1) = mean(y(mask)) / mean(u(mask)); %#ok<AGROW>
            inputAll  = [inputAll;  u(mask)];             %#ok<AGROW>
            outputAll = [outputAll; y(mask)];             %#ok<AGROW>
        end
        if ~isempty(gains), gainPerSet(end+1) = mean(gains); end %#ok<AGROW>
    end
    if isempty(gainPerSet)
        gainAvg = NaN;  affine = struct('gain', NaN, 'offset', NaN);
        fprintf('No settled window anywhere - static gain not measured.\n');
        return
    end
    gainAvg = mean(gainPerSet);
    coeff   = [inputAll, ones(numel(inputAll),1)] \ outputAll;
    affine  = struct('gain', coeff(1), 'offset', coeff(2));
    fprintf('Static gain %.8f (%+.3f mHz at 50 Hz)\n', gainAvg, (gainAvg-1)*50e3);
end

function mask = settledMask(u, nSettle)
    % A moving range, not a derivative: the input is smooth but never exactly
    % constant, so the tolerance comes from the quietest window this record manages.
    span = movmax(u, [nSettle-1, 0]) - movmin(u, [nSettle-1, 0]);
    span(1:min(nSettle-1, numel(span))) = Inf;
    if ~any(isfinite(span)), mask = false(size(u)); return; end
    mask = span <= max(1e-3*(max(u) - min(u)), 2*min(span));
end

function candidates = buildCandidates(runLin, runHW, runNL, poleGrid, naRange, ...
    nbRange, hwNb, hwNf, hwNames, hwCtors, nlNames, nlCtors)
    % One flat list, swept in order: LINEAR, then HW, then NLARX.
    list = {};
    if runLin
        for nPoles = poleGrid
            for nZeros = 0:(nPoles-1)
                list{end+1} = struct('type','LINEAR', 'order',[nPoles nZeros 0], ...
                    'nl', sprintf('tfest_%dp%dz', nPoles, nZeros), 'ctors', {{}}); %#ok<AGROW>
            end
        end
    end
    if runHW
        for nb = hwNb
            for nf = hwNf
                for i = 1:numel(hwNames)
                    for j = 1:numel(hwNames)
                        list{end+1} = struct('type','HW', 'order',[nb nf 0], ...
                            'nl', sprintf('%s-in/%s-out', hwNames{i}, hwNames{j}), ...
                            'ctors', {{hwCtors{i}, hwCtors{j}}}); %#ok<AGROW>
                    end
                end
            end
        end
    end
    if runNL
        for na = naRange
            for nb = nbRange
                for i = 1:numel(nlNames)
                    list{end+1} = struct('type','NLARX', 'order',[na nb 0], ...
                        'nl', nlNames{i}, 'ctors', {nlCtors(i)}); %#ok<AGROW>
                end
            end
        end
    end
    candidates = [list{:}];
end

function opts = estOptions(maxIter, tol, tfestTol)
    opts.lin = tfestOptions('SearchMethod','lm');
    opts.lin.SearchOptions.MaxIterations = maxIter;
    opts.lin.SearchOptions.Tolerance = tfestTol;
    opts.hw = nlhwOptions('SearchMethod','lm');
    opts.hw.SearchOptions.MaxIterations = maxIter;
    opts.hw.SearchOptions.Tolerance = tol;
    opts.nlarx = nlarxOptions('SearchMethod','lm');
    opts.nlarx.SearchOptions.MaxIterations = maxIter;
    opts.nlarx.SearchOptions.Tolerance = tol;
end

function results = runSearch(candidates, dTrain, dVal, opts, keepFit, targetFit, Ts)
    % Serial, so the grid order is the search order. Stops at the first candidate
    % clearing targetFit on both train and val. Every candidate prints a line -
    % keep, drop or error - so the window shows the whole sweep, not just the
    % survivors. Val is still simulated only for candidates that cleared train,
    % so one dropped on train reports val as '-' rather than paying for it.
    kept = {};  nTried = 0;  nDrop = 0;  nErr = 0;
    for k = 1:numel(candidates)
        cand = candidates(k);
        nTried = nTried + 1;
        trainFit = NaN;  valFit = NaN;  status = 'keep';  note = '';
        try
            sys = estimateModel(cand, dTrain, opts, Ts);
            trainFit = fitOf(sys, dTrain);
            if ~isfinite(trainFit) || trainFit < keepFit
                status = 'drop';
            else
                valFit = fitOf(sys, dVal);    % the second simulation, train survivors only
                if ~isfinite(valFit) || valFit < keepFit, status = 'drop'; end
            end
        catch err
            status = 'error';
            note = err.identifier;
            if isempty(note), note = err.message; end
        end
        reportCandidate(status, cand, trainFit, valFit, note);
        if strcmp(status, 'drop')
            nDrop = nDrop + 1;  continue
        elseif strcmp(status, 'error')
            nErr = nErr + 1;  continue
        end
        kept{end+1} = struct('type',cand.type, 'order',cand.order, 'nl',cand.nl, ...
            'trainFit',trainFit, 'valFit',valFit, 'sys',sys, 'combo',k); %#ok<AGROW>
        if trainFit >= targetFit && valFit >= targetFit, break; end
    end
    results = [kept{:}];
    fprintf('%d of %d candidates tried: %d kept (>= %g%% on train and val), %d below, %d errored.\n', ...
        nTried, numel(candidates), numel(kept), keepFit, nDrop, nErr);
end

function reportCandidate(status, cand, trainFit, valFit, note)
    % One line per candidate, fixed columns so the sweep reads as a table and the
    % status tag stays greppable out of a diary.
    if ~isempty(note)
        note = strtrim(regexprep(note, '\s+', ' '));
        if numel(note) > 60, note = [note(1:57) '...']; end
        note = ['   ' note];
    end
    fprintf('  %-5s %-6s [%-10s] %-22s train %s  val %s%s\n', status, cand.type, ...
        num2str(cand.order), cand.nl, fitStr(trainFit), fitStr(valFit), note);
end

function s = fitStr(v)
    % Same width either way, so the columns line up whether or not val was run.
    if isfinite(v), s = sprintf('%6.2f%%', v); else, s = sprintf('%7s', '-'); end
end

function sys = estimateModel(cand, data, opts, Ts)
    switch cand.type
        case 'LINEAR'   % no Ts argument: tfest returns a continuous model
            sys = tfest(data, cand.order(1), cand.order(2), cand.order(3)*Ts, opts.lin);
        case 'HW'
            sys = nlhw(data, cand.order, cand.ctors{1}(), cand.ctors{2}(), opts.hw);
        case 'NLARX'
            sys = nlarx(data, cand.order, cand.ctors{1}(), opts.nlarx);
    end
end

function cand = refineCandidate(cand, candidates, dTrain, dVal, optRefine, Ts)
    % Tight refit, adopted only if train improves.
    try
        sys = estimateModel(candidates(cand.combo), dTrain, optRefine, Ts);
        trainFit = fitOf(sys, dTrain);
        if isfinite(trainFit) && trainFit > cand.trainFit
            cand.sys = sys;  cand.trainFit = trainFit;  cand.valFit = fitOf(sys, dVal);
        end
    catch
    end
end

function fitPct = fitOf(sys, data)
    % Horizon Inf is pure simulation; NRMSE %, averaged over experiments.
    [~, fitPct] = compare(data, sys, Inf);
    if iscell(fitPct), fitPct = cell2mat(fitPct); end
    fitPct = mean(fitPct);
end

function score = scoreWithDelay(sys, stepRaw, nk, isTrain)
    % A pure delay commutes with any time-invariant system, so delaying u and
    % scoring the winner is the series (delay -> model). No product object needed.
    nSets = numel(stepRaw);
    score = struct('perSet', nan(1,nSets), 'tests', zeros(1,nSets), ...
        'train', NaN, 'val', NaN, 'all', NaN);
    for s = 1:nSets
        if isempty(stepRaw{s}), continue; end
        score.tests(s) = numExp(stepRaw{s});
        try
            score.perSet(s) = fitOf(sys, delayInput(stepRaw{s}, nk));
        catch
        end
    end
    ok = isfinite(score.perSet) & score.tests > 0;
    weighted = @(m) sum(score.perSet(m) .* score.tests(m)) / sum(score.tests(m));
    if any(ok),            score.all   = weighted(ok);            end
    if any(ok & isTrain),  score.train = weighted(ok & isTrain);  end
    if any(ok & ~isTrain), score.val   = weighted(ok & ~isTrain); end
end

function [sysDelay, sysFinal] = delayModel(sys, nkSamples, Ts)
    % The lag as its own factor: seconds for a continuous model, samples for a
    % discrete one. idnlhw/idnlarx cannot be multiplied at all.
    sysDelay = [];  sysFinal = sys;
    if nkSamples <= 0, return; end
    modelTs = 0;
    try
        modelTs = sys.Ts;
    catch
    end
    if modelTs > 0
        sysDelay = idtf(1, 1, modelTs, 'IODelay', nkSamples);
    else
        sysDelay = idtf(1, 1, 'IODelay', nkSamples*Ts);
    end
    try
        sysFinal = sys * sysDelay;
    catch
        sysFinal = sys;
    end
end

function [dcGain, sysUnit] = unityDcForm(sys)
    % G(s) = K * Gn(s) with Gn(0) = 1 exactly, so the static gain and the dynamics
    % travel as separate numbers and neither is lost to the other's rounding.
    dcGain = NaN;  sysUnit = [];
    try
        G = tf(sys);  dcGain = dcgain(G);
    catch
        return          % idnlarx / idnlhw have no transfer function at all
    end
    if ~isscalar(dcGain) || ~isfinite(dcGain) || dcGain == 0, dcGain = NaN; return; end
    [num, den] = tfdata(G, 'v');     % scale the numerator, so Gn stays monic
    sysUnit = tf(num/dcGain, den);
end

function plotSets(sets, names, isTrain, sys, drawSets, titleStr, subStr, pngPath)
    % One tile per excitation, every repeat drawn. With sys empty this is the data
    % overview, coloured by train/val; with a model it is the fit, measured in grey,
    % the model in red and the input blue. The model goes on last - with no delay
    % in play it hides under the input otherwise.
    trainCol = [0.20 0.35 0.75];   % blue
    valCol   = [0.85 0.33 0.10];   % orange
    % Black on the overview, where blue already means "train"; the house blue on the
    % fit figures, where measured is grey and the model red, so nothing collides.
    if isempty(sys), inputCol = [0 0 0]; else, inputCol = trainCol; end
    nCols = ceil(sqrt(numel(names)));  nRows = ceil(numel(names)/nCols);
    fig = figure('Name', titleStr, ...
        'Position', [60 60 min(1800, 430*nCols), min(1150, 310*nRows)]);
    layout = tiledlayout(fig, nRows, nCols, 'TileSpacing','compact', 'Padding','compact');
    legendAx = [];  legendH = [];  legendLabels = {};
    for s = 1:numel(names)
        ax = nexttile(layout);  hold(ax,'on');  grid(ax,'on');
        nRep = numExp(sets{s});
        if nRep == 0
            title(ax, sprintf('%s - no data', names{s}), 'FontSize',10, 'Interpreter','none');
            continue
        end
        if ~isempty(sys),      measCol = [0.35 0.35 0.35 0.40];
        elseif isTrain(s),     measCol = trainCol;
        else,                  measCol = valCol;
        end
        fits = nan(1, nRep);  simTraces = cell(1, nRep);  hMeas = gobjects(0);
        for e = 1:nRep
            rec = getexp(sets{s}, e);
            t = (0:numel(rec.y)-1)' * sampleTime(rec);
            hMeas(end+1) = plot(ax, t, rec.y, '-', 'Color', measCol, 'LineWidth', 0.7); %#ok<AGROW>
            if isempty(sys), continue; end
            try
                [ySim, fitRep] = compare(rec, sys, Inf);
                if iscell(fitRep), fitRep = cell2mat(fitRep); end
                fits(e) = mean(fitRep);  simTraces{e} = {t, ySim.y};
            catch
            end
        end
        recDraw = getexp(drawSets{s}, 1);  u = recDraw.u(:);
        hInput = plot(ax, (0:numel(u)-1)' * sampleTime(recDraw), u, '-', ...
            'Color', inputCol, 'LineWidth', 1.5);
        hModel = gobjects(0);
        for e = 1:nRep
            if isempty(simTraces{e}), continue; end
            hModel(end+1) = plot(ax, simTraces{e}{1}, simTraces{e}{2}, 'r-', ...
                'LineWidth', 1.5); %#ok<AGROW>
        end
        if isempty(sys)
            title(ax, sprintf('%s  (%d tests)', names{s}, nRep), ...
                'FontSize',10, 'Interpreter','none');
        else
            title(ax, {names{s}, sprintf('FIT %.1f%% (%.1f-%.1f) - %d tests', ...
                mean(fits,'omitnan'), min(fits), max(fits), nRep)}, ...
                'FontSize',10, 'Interpreter','none');
        end
        xlabel(ax,'time (s)');  ylabel(ax,'Hz');
        if isempty(legendAx) && (isempty(sys) || ~isempty(hModel))
            % Proxy lines, so the overview legend can name both set colours even
            % though this tile only draws one of them.
            legendAx = ax;
            if isempty(sys)
                legendH = [plot(ax, NaN, NaN, '-', 'Color', trainCol, 'LineWidth', 1.5), ...
                           plot(ax, NaN, NaN, '-', 'Color', valCol,   'LineWidth', 1.5), ...
                           hInput];
                legendLabels = {'train (blue)', 'validation (orange)', ...
                                'simulation input (black)'};
            else
                legendH = [hMeas(1), hModel(1), hInput];
                legendLabels = {'measured, all tests (grey)', ...
                                'model, per test (red)', 'simulation input (blue)'};
            end
        end
    end
    if isempty(subStr)
        title(layout, titleStr, 'Interpreter','none', 'FontSize',12, 'FontWeight','bold');
    else
        title(layout, {titleStr; subStr}, 'Interpreter','none', 'FontSize',12, ...
            'FontWeight','bold');
    end
    if ~isempty(legendH)
        lg = legend(legendAx, legendH, legendLabels, 'FontSize',8, 'Box','off', ...
            'Orientation','horizontal');
        lg.Layout.Tile = 'north';
    end
    if ~isempty(pngPath)
        try
            exportgraphics(fig, pngPath, 'Resolution', 150);
        catch
        end
    end
end

function pngPath = pngName(doSave, tag)
    if doSave, pngPath = [regexprep(tag, '[^\w\-]', '_') '.png']; else, pngPath = ''; end
end

function Ts = sampleTime(data)
    Ts = data.Ts;
    if iscell(Ts), Ts = Ts{1}; end
    Ts = double(Ts);
end

function n = numExp(data)
    if isempty(data), n = 0; else, [~,~,~,n] = size(data); end
end
