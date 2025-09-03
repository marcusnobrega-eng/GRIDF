%%%%%%%%%%%%%%%%%%%%%%%%
% ONLY CETESB + STATION

%% =======================================================================
%  Master IDF Builder — all datasets × both disaggregation modes
%  Author: Maria / Marcus (HydroBR)
%  Purpose: Fit Sherman IDF (K,a,b,c) using 24h AMD stacks + either:
%           (A) Local disaggregation rasters (Depth(D)/Depth(24h)), or
%           (B) CETESB fixed factors
%  Outputs (per dataset/method): single-band GeoTIFFs, a stacked GeoTIFF
%  Notes:
%   - Assumes all AMD GeoTIFFs are already in mm/day (scaled), so
%     APPLY_SCALE_OFFSET = false.
%   - Saves to OUT_ROOT/<DATASET>/<DISAGG>/...
%   - Robust to small grid misalignments via regridToBase()
% ========================================================================

clear; clc;

% --- Station-based disaggregation (global for all datasets) ---
USE_STATION_DISAGG = true;   % <- turn ON to use nearest “ok-fit” station
STATION_CSV = 'C:\Users\marcu\OneDrive - University of Arizona\Papers\Rainfall_Distribution_Paper\RainfallAnalysis\Subhourly_Disag_Log.csv';


%% ====================== GLOBAL USER SETTINGS ==========================
RETURN_PERIODS_YR = [2 5 10 25 50 75 100];
DURATIONS_MIN     = [5,10,15,20,25,30,60,360,480,600,720,1440];
MIN_YEARS_REQ     = 5;

% Global defaults (used if a dataset doesn't override)
DEFAULT_YEAR_START = 1994;
DEFAULT_YEAR_END   = 2024;


APPLY_SCALE_OFFSET = false;     % all tiffs already scaled -> leave false
DWGD_SCALE  = 0.00686666;       % (kept for completeness if ever needed)
DWGD_OFFSET = 225.0;

% CETESB factors (depth(D)/depth(24h)); 24h normalized to 1 by default
CETESB_TABLE = containers.Map( ...
    [5 10 15 20 25 30 60 360 480 600 720 1440], ...
    [0.12 0.191 0.248 0.287 0.322 0.354 0.479 0.821 0.889 0.935 0.969 1.14] );
CETESB_NORMALIZE_24H_TO_ONE = true;

% Local disaggregation rasters (IDW set)
IDW_DIR = 'C:\Users\marcu\OneDrive - University of Arizona\Papers\Rainfall_Distribution_Paper\RainfallAnalysis\IDW_Rasters\';
USE_K   = 'k10';   % or 'k12'
fnPat   = @(lab) fullfile(IDW_DIR, sprintf('IDW_P%s_Pday_res0.100_%s_p2.0.tif', lab, USE_K));

% Output root
OUT_ROOT = 'C:\Users\marcu\OneDrive - University of Arizona\Papers\Rainfall_Distribution_Paper\IDF_params_out_MASTER\';

% Gumbel KS options
DO_GUMBEL_KS   = true;
ALPHA_KS       = 0.05;

% Type casting to save RAM
CAST_SINGLE    = true;

% Disaggregation modes to run

DISAGG_MODES = {'RASTER', 'CETESB', 'STATION'};


% ====================== DATASETS ======================
% Add per-dataset year ranges here (falls back to DEFAULT_* if missing)
DATASETS = struct( ...
    'name',        {'XAVIER','IMERG','PERSIANN','CHIRPS_BC'}, ...
    'AMD_DIRS',    { ...
    {'C:\Users\marcu\OneDrive\Documentos\GitHub\Rainfall_Temp_Distributor\Data\BiasCorrected\'} ...
    {'G:\My Drive\IMERG_Max\IMERG_BiasCorrected_AMaxDaily\'}, ...
    {'G:\My Drive\PERSIANN_Max\PERSIANN_BiasCorrected_AMaxDaily\'}, ...
    {'G:\My Drive\CHRIPS_Max\CHIRPS_BiasCorrected_AMaxDaily\'} ...
    }, ...
    'AMD_PATTERN', { ...
    'BR_DWGD_prmax_*.tif', ...
    'IMERG_MaxDaily_0p1deg_*_Brazil_BC.tif', ...
    'PERSIANN_MaxDaily_0p25deg_*_Brazil_BC.tif', ...
    'CHIRPS_MaxDaily_0p1deg_*_Brazil_BC.tif' ...
    }, ...
    'RECURSIVE',   {false,false,false,false}, ...
    'year_start',  {1994, 2000, 1994, 1994}, ...   % <-- IMERG starts at 2000
    'year_end',    {2024, 2024, 2024, 2024} ...
    );
% Column names expected in station CSV: all_c5, all_c10, ..., all_c1440
STATION_DUR_COLS = arrayfun(@(d) sprintf('all_c%d', d), DURATIONS_MIN, 'uni', false);

% ============ Build the DISAGG_RASTERS map for 'RASTER' mode ===========
DISAGG_RASTERS_TEMPLATE = containers.Map('KeyType','double','ValueType','char');
for j = 1:numel(DURATIONS_MIN)
    Dm = DURATIONS_MIN(j);
    if Dm == 1440
        DISAGG_RASTERS_TEMPLATE(Dm) = '';  % 24h => ratio = 1 (no file)
    else
        if Dm < 60, lab = sprintf('%dm', Dm); else, lab = sprintf('%dh', round(Dm/60)); end
        DISAGG_RASTERS_TEMPLATE(Dm) = fnPat(lab);
    end
end

% Sanity check files (except 24h) — only if RASTER mode will be run
if any(strcmpi(DISAGG_MODES,'RASTER'))
    for j = 1:numel(DURATIONS_MIN)
        Dm = DURATIONS_MIN(j);
        if Dm == 1440, continue; end
        pth = DISAGG_RASTERS_TEMPLATE(Dm);
        assert(~isempty(pth) && isfile(pth), 'Missing disagg raster for %g min: %s', Dm, pth);
    end
end

%% ====================== RUN ALL COMBINATIONS =========================
for dd = 1:numel(DATASETS)
    ds = DATASETS(dd);

    for mm = 1:numel(DISAGG_MODES)
        modeName = DISAGG_MODES{mm};   % 'RASTER' or 'CETESB'
        outDir = fullfile(OUT_ROOT, ds.name, modeName);
        if ~exist(outDir,'dir'), mkdir(outDir); end

        fprintf('\n=== Dataset: %s | Disaggregation: %s ===\n', ds.name, modeName);

        % ----- Gather AMD files for this dataset -----
        files = listGeoTiffs(ds.AMD_DIRS, ds.AMD_PATTERN, ds.RECURSIVE);

        % Per-dataset year window (fallback to defaults)
        yrStart = DEFAULT_YEAR_START;
        yrEnd   = DEFAULT_YEAR_END;
        if isfield(ds,'year_start') && ~isempty(ds.year_start), yrStart = ds.year_start; end
        if isfield(ds,'year_end')   && ~isempty(ds.year_end),   yrEnd   = ds.year_end;   end
        fprintf('Using year window %d–%d for %s\n', yrStart, yrEnd, ds.name);

        % Extract trailing 4-digit year: ..._YYYY.tif (matches last _YYYY)
        rxYearEnd = '_(\d{4})(?=\D*$)';
        keep = false(numel(files),1); years = nan(numel(files),1);
        for i=1:numel(files)
            t = regexp(files(i).name, rxYearEnd, 'tokens', 'once');
            if ~isempty(t)
                yy = str2double(t{1});
                if yy>=yrStart && yy<=yrEnd
                    keep(i) = true; years(i) = yy;
                end
            end
        end
        files = files(keep); years = years(keep);
        assert(~isempty(files), 'No AMD rasters found for %s in %d–%d.', ds.name, yrStart, yrEnd);


        % Deduplicate by year
        [uy, ia] = unique(years, 'stable');
        if numel(uy) < numel(years)
            fprintf('Warning: duplicates detected; keeping first per year.\n');
        end
        years = uy; files = files(ia);
        [years, ord] = sort(years); files = files(ord);
        fprintf('Years found: %s\n', sprintf('%d ', years));

        % Base grid
        [firstA, Rgeo] = readAMD(fullfile(files(1).folder, files(1).name), APPLY_SCALE_OFFSET, DWGD_SCALE, DWGD_OFFSET);
        baseInfo = georasterinfo(fullfile(files(1).folder, files(1).name));
        assert(isGeographicRef(Rgeo), 'AMD rasters must be geographic (EPSG:4326).');

        [nrows, ncols] = size(firstA);
        nYears = numel(files);

        AMD_all = nan(nrows, ncols, nYears, 'single');
        AMD_all(:,:,1) = single(firstA);

        % Welford streaming stats
        AMD_mean = zeros(nrows,ncols,'double');
        AMD_M2   = zeros(nrows,ncols,'double');
        AMD_cnt  = zeros(nrows,ncols,'double');
        [AMD_mean, AMD_M2, AMD_cnt] = welfordUpdate(AMD_mean, AMD_M2, AMD_cnt, firstA);

        for i=2:numel(files)
            [Ai, Ri] = readAMD(fullfile(files(i).folder, files(i).name), APPLY_SCALE_OFFSET, DWGD_SCALE, DWGD_OFFSET);
            Ai = regridToBase(Ai, Ri, Rgeo, files(i).name);   % align to base
            [AMD_mean, AMD_M2, AMD_cnt] = welfordUpdate(AMD_mean, AMD_M2, AMD_cnt, Ai);
            AMD_all(:,:,i) = single(Ai);
            if mod(i, max(1,round(numel(files)/20)))==0
                fprintf('Accumulated %d/%d years...\n', i, numel(files));
            end
        end

        AMD_var = AMD_M2 ./ max(AMD_cnt-1, 1);
        AMD_std = sqrt(AMD_var);
        mask_bad = AMD_cnt < MIN_YEARS_REQ;
        AMD_mean(mask_bad) = NaN;
        AMD_std(mask_bad)  = NaN;

        if CAST_SINGLE
            AMD_mean = single(AMD_mean);
            AMD_std  = single(AMD_std);
        end

        % KS Gumbel goodness-of-fit (per pixel)
        KS_D_img   = nan(nrows,ncols,'single');
        KS_p_img   = nan(nrows,ncols,'single');
        KS_reject  = zeros(nrows,ncols,'uint8');
        if DO_GUMBEL_KS
            totalPix = nrows*ncols;
            for p = 1:totalPix
                if mod(p, max(1,round(totalPix*0.02)))==0, fprintf('KS: %5.1f%%\n',100*p/totalPix); end
                [r, c] = ind2sub([nrows,ncols], p);
                x = double(squeeze(AMD_all(r,c,:)));
                x = x(isfinite(x));
                if numel(x) < MIN_YEARS_REQ, continue; end
                [Dks, pval] = ks_gumbel_moments(x);
                KS_D_img(p)  = single(Dks);
                KS_p_img(p)  = single(pval);
                KS_reject(p) = uint8(pval < ALPHA_KS);
            end
        end

        % -------- Disaggregation ratios RATIO(:,:,j) = Depth(Dj)/Depth(24h) ------
        nDur    = numel(DURATIONS_MIN);
        D_hours = single(DURATIONS_MIN(:)'/60);

        switch upper(modeName)
            case 'STATION'
                assert(USE_STATION_DISAGG, 'STATION mode requires USE_STATION_DISAGG=true.');
                Tst = readtable(STATION_CSV);

                % Verify columns
                assert(all(ismember(STATION_DUR_COLS, Tst.Properties.VariableNames)), ...
                    'Station CSV missing required columns: %s', strjoin(STATION_DUR_COLS, ', '));

                % Keep only “ok-fit” stations with complete ratios
                hasOk   = contains(lower(string(Tst.note)), 'ok');
                hasAll  = all(isfinite(Tst{:, STATION_DUR_COLS}), 2);
                keepSt  = hasOk & hasAll;
                assert(any(keepSt), 'No stations with ok-fit and complete ratios found in CSV.');

                latS    = double(Tst.latitude(keepSt));
                lonS    = double(Tst.longitude(keepSt));
                ratioS  = double(Tst{keepSt, STATION_DUR_COLS});  % Nst x Ndur

                % Clean & enforce monotonicity per station
                ratioS(ratioS<=0) = NaN;
                ratioS(:, end)    = 1;                            % 24h ratio = 1
                for iSt = 1:size(ratioS,1)
                    v = ratioS(iSt,:);
                    if any(~isfinite(v)), continue; end
                    ratioS(iSt,:) = min(1, cummax(v));            % non-decreasing, capped at 1
                end

                % Rasterize by nearest station (no smoothing)
                [latq, lonq] = baseCellCenterGrid(Rgeo);
                RATIO = nan(nrows,ncols,nDur, 'single');

                % Optional diagnostic: which station was chosen
                stIdx = (1:numel(latS))';
                Fidx  = scatteredInterpolant(lonS, latS, double(stIdx), 'nearest', 'nearest');
                NEAREST_STATION_IDX = single(Fidx(lonq, latq));   %#ok<NASGU>

                for j = 1:nDur
                    Fj = scatteredInterpolant(lonS, latS, ratioS(:,j), 'nearest', 'nearest');
                    RATIO(:,:,j) = single(Fj(lonq, latq));
                end

                % Dummy QC placeholders so later writes don’t fail
                viol_cnt = zeros(nrows,ncols,'single'); %#ok<NASGU>
                viol_mag = zeros(nrows,ncols,'single'); %#ok<NASGU>
                isProblem = false(nrows,ncols);         %#ok<NASGU>
                srcR = zeros(nrows,ncols,'single');     %#ok<NASGU>
                srcC = zeros(nrows,ncols,'single');     %#ok<NASGU>
                srcDist = zeros(nrows,ncols,'single');  %#ok<NASGU>

            case 'CETESB'
                % Build a single ratio vector, replicate to all pixels
                ratioVec = zeros(1,nDur);
                for j = 1:nDur
                    ratioVec(j) = CETESB_TABLE(DURATIONS_MIN(j));
                end
                if CETESB_NORMALIZE_24H_TO_ONE
                    ratioVec = ratioVec ./ ratioVec(end); % force 24h=1
                end
                ratioVec(end) = 1;
                ratioVec = min(1, cummax(ratioVec));      % monotone & <=1

                RATIO = repmat(single(reshape(ratioVec,1,1,[])), nrows, ncols);

                % QC placeholders
                viol_cnt = zeros(nrows,ncols,'single'); %#ok<NASGU>
                viol_mag = zeros(nrows,ncols,'single'); %#ok<NASGU>
                isProblem = false(nrows,ncols);         %#ok<NASGU>
                srcR = zeros(nrows,ncols,'single');     %#ok<NASGU>
                srcC = zeros(nrows,ncols,'single');     %#ok<NASGU>
                srcDist = zeros(nrows,ncols,'single');  %#ok<NASGU>

            case 'RASTER'
                % Read each disaggregation raster, regrid to base
                RATIO = nan(nrows,ncols,nDur,'single');
                for j = 1:nDur
                    Dm = DURATIONS_MIN(j);
                    if Dm == 1440
                        RATIO(:,:,j) = 1;    % 24h ratio = 1
                        continue;
                    end
                    pth = DISAGG_RASTERS_TEMPLATE(Dm);
                    [Rj, Rjref] = readAMD(pth, false, DWGD_SCALE, DWGD_OFFSET);
                    Rj = regridToBase(Rj, Rjref, Rgeo, pth);
                    RATIO(:,:,j) = single(Rj);
                end

                % Repair: enforce monotonicity (cap at 1.0) & then fill bad pixels
                [RATIO, violCnt, violMag, ~, ~] = enforceRatioMonotonicity(RATIO, DURATIONS_MIN, 1e-6, 1.0);

                % A "problem" pixel is any with NaN anywhere or any violation
                isFiniteAll = all(isfinite(RATIO), 3);
                isProblem   = ~isFiniteAll | (violCnt > 0);

                % Copy entire ratio vector from nearest clean pixel (euclidean in row/col)
                isClean = ~isProblem;
                [RATIO, srcR, srcC, srcDist] = fillRatiosFromNearestClean(RATIO, isClean);

                % For QC outputs expected later
                viol_cnt = single(violCnt); %#ok<NASGU>
                viol_mag = single(violMag); %#ok<NASGU>

            otherwise
                error('Unknown DISAGG mode: %s', modeName);
        end

        % Convert ratios to intensity scale factor S(t) = ratio / (t/60)
        S = RATIO ./ reshape(D_hours, [1 1 nDur]);



        % Precompute Gumbel reduced variate
        yT = -log(log(RETURN_PERIODS_YR ./ (RETURN_PERIODS_YR - 1)));
        if CAST_SINGLE, yT = single(yT); end

        % Allocate outputs
        k_img    = nan(nrows,ncols,'single');
        a_img    = nan(nrows,ncols,'single');
        b_img    = nan(nrows,ncols,'single');
        c_img    = nan(nrows,ncols,'single');
        R2_img   = nan(nrows,ncols,'single');
        RMSE_img = nan(nrows,ncols,'single');
        MSE_img  = nan(nrows,ncols,'single');

        % ====================== Fit Sherman per pixel ===================
        totalPix = nrows*ncols;
        for p = 1:totalPix
            if mod(p, max(1,round(totalPix*0.02)))==0
                fprintf('Fit: %5.1f%%\n', 100*p/totalPix);
            end
            if ~isfinite(AMD_mean(p)) || ~isfinite(AMD_std(p)) || AMD_std(p)==0
                continue;
            end

            % 24h quantiles Q(T)
            Q = double(AMD_mean(p)) + double(AMD_std(p)) * (double(yT)*(1/1.282) - 0.450047);
            if ~all(isfinite(Q)), continue; end

            [r,c] = ind2sub([nrows,ncols], p);
            S_pix = squeeze(S(r,c,:));
            validDur = isfinite(S_pix) & S_pix>0;
            if nnz(validDur) < 3, continue; end

            Igrid = Q(:) * double(S_pix(validDur))';  % nRP x nDur_valid
            [TT, DD] = ndgrid(RETURN_PERIODS_YR, DURATIONS_MIN(validDur));
            y = log10(Igrid(:)); good = isfinite(y);
            if nnz(good) < 3, continue; end

            TTg = TT(:); DDg = DD(:); yg = y(good);

            % Linear Bernard warm start
            Xlin  = [ones(numel(y),1), log10(TT(:)), -log10(DD(:))];
            beta0 = Xlin(good,:) \ yg;
            K0 = 10.^beta0(1);  a0 = beta0(2);  c0 = max(0, -beta0(3));
            b0 = 0.1 * min(DURATIONS_MIN(validDur));

            % Nonlinear LS in log space
            funLog = @(th,x) log10(th(1)) + th(2).*log10(x(:,1)) - th(4).*log10(th(3) + x(:,2));
            lb = [eps, -1, 0, eps];
            ub = [Inf,  1,  5*max(DDg), 5];
            xdata = [TTg(good), DDg(good)]; ydata = yg;

            if exist('lsqcurvefit','file')
                opts = optimoptions('lsqcurvefit','Display','off');
                th = lsqcurvefit(funLog, [K0 a0 b0 c0], xdata, ydata, lb, ub, opts);
            else
                toTheta = @(w) [exp(w(1)), w(2), exp(w(3))-1, exp(w(4))-1];
                obj = @(w) nansum( ( ydata - funLog(toTheta(w), xdata) ).^2 );
                w0  = [log(K0), a0, log1p(b0), log1p(c0)];
                w   = fminsearch(obj, w0, optimset('Display','off'));
                th  = toTheta(w);
            end

            K = th(1); a = th(2); b = th(3); cpar = th(4);

            Ihat = (K*(TT.^a)) ./ ((b + DD).^cpar);
            res  = Igrid(:) - Ihat(:);
            MSE  = mean(res.^2,'omitnan'); RMSE = sqrt(MSE);
            SSres = nansum((Igrid(:)-Ihat(:)).^2);
            SStot = nansum((Igrid(:)-mean(Igrid(:),'omitnan')).^2);
            R2   = 1 - SSres/SStot;
            
            k_img(p)    = single(K);
            a_img(p)    = single(a);
            b_img(p)    = single(b);
            c_img(p)    = single(cpar);
            R2_img(p)   = single(R2);
            RMSE_img(p) = single(RMSE);
            MSE_img(p)  = single(MSE);
        end

        % -------- Fill failed pixels from nearest successful fit (simple fallback) --------
        inDomain = AMD_mean > 0;
        bands = cat(3, k_img, a_img, b_img, c_img, R2_img, RMSE_img, MSE_img);  % add more if you like
        keyOK = isfinite(k_img) & isfinite(a_img) & isfinite(b_img) & isfinite(c_img) & isfinite(R2_img) & inDomain;
        
       
        [bands_filled, srcR, srcC, distPx] = fillFromNearestValid(bands, keyOK);
        bands_filled(repmat(~inDomain,1,1,3)) = nan;

        % Unpack back
        k_img    = bands_filled(:,:,1);
        a_img    = bands_filled(:,:,2);
        b_img    = bands_filled(:,:,3);
        c_img    = bands_filled(:,:,4);
        R2_img   = bands_filled(:,:,5);
        RMSE_img = bands_filled(:,:,6);
        MSE_img  = bands_filled(:,:,7);
        
        % (Optional) write diagnostics later with your QC outputs


        % ====================== WRITE OUTPUTS ==========================
        srsText = crsTextFromInfo(baseInfo);

        OUT_K_TIF    = fullfile(outDir,'IDF_k.tif');
        OUT_A_TIF    = fullfile(outDir,'IDF_a.tif');
        OUT_B_TIF    = fullfile(outDir,'IDF_b.tif');
        OUT_C_TIF    = fullfile(outDir,'IDF_c.tif');
        OUT_R2_TIF   = fullfile(outDir,'IDF_R2.tif');
        OUT_RMSE_TIF = fullfile(outDir,'IDF_RMSE.tif');
        OUT_MSE_TIF  = fullfile(outDir,'IDF_MSE.tif');

        gtw(OUT_K_TIF,    k_img,  Rgeo, baseInfo);
        gtw(OUT_A_TIF,    a_img,  Rgeo, baseInfo);
        gtw(OUT_B_TIF,    b_img,  Rgeo, baseInfo);
        gtw(OUT_C_TIF,    c_img,  Rgeo, baseInfo);
        gtw(OUT_R2_TIF,   R2_img, Rgeo, baseInfo);
        gtw(OUT_RMSE_TIF, RMSE_img, Rgeo, baseInfo);
        gtw(OUT_MSE_TIF,  MSE_img,  Rgeo, baseInfo);

        % QC outputs for RASTER mode (includes nearest-source diagnostics)
        if strcmpi(modeName,'RASTER')
            OUT_VIOLCNT_TIF = fullfile(outDir,'QC_disagg_violation_count.tif');
            OUT_VIOLMAG_TIF = fullfile(outDir,'QC_disagg_violation_magnitude.tif');
            OUT_PROB_TIF    = fullfile(outDir,'QC_problem_mask.tif');
            OUT_SRCROW_TIF  = fullfile(outDir,'QC_nearest_src_row.tif');
            OUT_SRCCOL_TIF  = fullfile(outDir,'QC_nearest_src_col.tif');
            OUT_SRCDIST_TIF = fullfile(outDir,'QC_nearest_distance_px.tif');

            gtw(OUT_VIOLCNT_TIF, single(viol_cnt),   Rgeo, baseInfo);
            gtw(OUT_VIOLMAG_TIF, single(viol_mag),   Rgeo, baseInfo);
            gtw(OUT_PROB_TIF,    uint8(isProblem),   Rgeo, baseInfo);
            gtw(OUT_SRCROW_TIF,  single(srcR),       Rgeo, baseInfo);
            gtw(OUT_SRCCOL_TIF,  single(srcC),       Rgeo, baseInfo);
            gtw(OUT_SRCDIST_TIF, single(srcDist),    Rgeo, baseInfo);
        end

        OUT_KS_D_TIF   = fullfile(outDir,'IDF_KS_D.tif');
        OUT_KS_P_TIF   = fullfile(outDir,'IDF_KS_p.tif');
        OUT_KS_REJ_TIF = fullfile(outDir,'IDF_KS_reject.tif');

        gtw(OUT_KS_D_TIF,   KS_D_img,  Rgeo, baseInfo);
        gtw(OUT_KS_P_TIF,   KS_p_img,  Rgeo, baseInfo);
        gtw(OUT_KS_REJ_TIF, KS_reject, Rgeo, baseInfo);

        % Multiband stack
        OUT_STACK_TIF = fullfile(outDir, 'IDF_params_stack.tif');
        stack = cat(3, k_img, a_img, b_img, c_img, R2_img, RMSE_img, MSE_img, ...
            KS_D_img, KS_p_img, single(KS_reject));
        bandNames = {'K','a','b','c','R2','RMSE','MSE','KS_D','KS_p','KS_reject'};

        AMD_mean_single = single(AMD_mean);
        AMD_std_single  = single(AMD_std);
        Nyears_single   = single(AMD_cnt);
        stack = cat(3, stack, AMD_mean_single, AMD_std_single, Nyears_single);
        bandNames = [bandNames, {'AMD_mean','AMD_std','Nyears'}];

        gtw(OUT_STACK_TIF, stack, Rgeo, baseInfo);
        makeVRTWithBandNames(OUT_STACK_TIF, bandNames, Rgeo, srsText);

        fprintf('\nSaved outputs to: %s\n', outDir);
    end
end

%% ====================== HELPERS =====================================

function S = listGeoTiffs(dirs, pattern, recursive)
S = struct('folder',{},'name',{},'date',{},'bytes',{},'isdir',{},'datenum',{}); %#ok<AGROW>
for i=1:numel(dirs)
    d = dirs{i};
    if recursive
        L = dir(fullfile(d, '**', pattern));
    else
        L = dir(fullfile(d, pattern));
    end
    L = L(~[L.isdir]);
    S = [S; L]; %#ok<AGROW>
end
end

function [A,R] = readAMD(path, applyScale, scaleFac, offsetVal)
[A,R] = readRasterWithNaN(path);
if applyScale
    A = A*scaleFac + offsetVal;
end
end

function [A,R] = readRasterWithNaN(tifPath)
[A,R] = readgeoraster(tifPath);
A = double(A);
info = georasterinfo(tifPath);
if isfield(info,'MissingDataIndicator') && ~isempty(info.MissingDataIndicator)
    A(ismember(A, info.MissingDataIndicator)) = NaN;
end
if isfield(info,'NoData') && ~isempty(info.NoData)
    A(A==info.NoData) = NaN;
end
A(A<-1e19 | A>1e19) = NaN;
end

function tf = isGeographicRef(R)
tf = isa(R,'map.rasterref.GeographicCellsReference') || ...
    isa(R,'map.rasterref.GeographicPostingsReference');
end

function [meanA, M2, cnt] = welfordUpdate(meanA, M2, cnt, A)
valid = isfinite(A);
cNew  = cnt + valid;
delta = zeros(size(A)); idx = valid & (cNew>0);
delta(idx) = A(idx) - meanA(idx);
meanNew    = meanA; meanNew(idx) = meanA(idx) + delta(idx)./cNew(idx);
delta2     = zeros(size(A)); delta2(idx) = A(idx) - meanNew(idx);
M2New      = M2; M2New(idx) = M2(idx) + delta(idx).*delta2(idx);
meanA = meanNew; M2 = M2New; cnt = cNew;
end

function gtw(outPath, img, R, baseInfo)
ttags = struct('Compression', Tiff.Compression.LZW);
args  = {'TiffTags', ttags};
epsg = tryGetEPSG(baseInfo);
if ~isempty(epsg)
    args = [args, {'CoordRefSysCode', epsg}];
elseif isfield(baseInfo,'GeoTIFFTags') && isfield(baseInfo.GeoTIFFTags,'GeoKeyDirectoryTag') ...
        && ~isempty(baseInfo.GeoTIFFTags.GeoKeyDirectoryTag)
    args = [args, {'GeoKeyDirectoryTag', baseInfo.GeoTIFFTags.GeoKeyDirectoryTag}];
end
geotiffwrite(outPath, img, R, args{:});
end

function epsg = tryGetEPSG(info)
% Try to extract an EPSG integer from georasterinfo output (new + old MATLAB)
epsg = [];
try
    % Newer georasterinfo objects expose a CoordinateReferenceSystem
    if isfield(info,'CoordinateReferenceSystem') && ~isempty(info.CoordinateReferenceSystem)
        crs = info.CoordinateReferenceSystem;

        % Direct EPSG on the top-level CRS object
        if isprop(crs,'Authority') && isprop(crs,'Code') && strcmpi(string(crs.Authority),'EPSG')
            epsg = double(crs.Code); return;
        end

        % Sometimes the EPSG is on the nested Projected/Geographic CRS
        if isprop(crs,'ProjectedCRS') && ~isempty(crs.ProjectedCRS) && isprop(crs.ProjectedCRS,'Code')
            epsg = double(crs.ProjectedCRS.Code); return;
        end
        if isprop(crs,'GeographicCRS') && ~isempty(crs.GeographicCRS) && isprop(crs.GeographicCRS,'Code')
            epsg = double(crs.GeographicCRS.Code); return;
        end

        % Older georasterinfo structs
    elseif isfield(info,'ProjectedCRS') && ~isempty(info.ProjectedCRS) && isfield(info.ProjectedCRS,'EPSGCode')
        epsg = double(info.ProjectedCRS.EPSGCode); return;

        % GeoTIFF key fallback (FIXED duplicated condition)
    elseif isfield(info,'GeoTIFFTags') && ...
            (isfield(info.GeoTIFFTags,'ProjectedCSTypeGeoKey') || isfield(info.GeoTIFFTags,'GeographicTypeGeoKey'))

        if isfield(info.GeoTIFFTags,'ProjectedCSTypeGeoKey') && ~isempty(info.GeoTIFFTags.ProjectedCSTypeGeoKey)
            epsg = double(info.GeoTIFFTags.ProjectedCSTypeGeoKey); return;
        end
        if isfield(info.GeoTIFFTags,'GeographicTypeGeoKey') && ~isempty(info.GeoTIFFTags.GeographicTypeGeoKey)
            epsg = double(info.GeoTIFFTags.GeographicTypeGeoKey); return;
        end
    end
catch
    % leave empty
end
end


function s = crsTextFromInfo(info)
s = '';
try
    if isfield(info,'CoordinateReferenceSystem') && ~isempty(info.CoordinateReferenceSystem)
        crs = info.CoordinateReferenceSystem;
        if isprop(crs,'Authority') && isprop(crs,'Code') && strcmpi(string(crs.Authority),'EPSG')
            s = sprintf('EPSG:%d', crs.Code);
        elseif isprop(crs,'WellKnownText') && ~isempty(crs.WellKnownText)
            s = char(crs.WellKnownText);
        end
    end
catch
end
if isempty(s), s = 'EPSG:4326'; end
end

function Aout = regridToBase(Ain, Rin, Rbase, name, method)
if nargin < 5, method = 'linear'; end
if isequal(Rin.RasterSize, Rbase.RasterSize)
    [Xc,Yc] = meshgrid(1:Rbase.RasterSize(2), 1:Rbase.RasterSize(1));
    try
        if isGeographicRef(Rbase) && isGeographicRef(Rin)
            [latq,lonq] = intrinsicToGeographic(Rbase, Xc, Yc);
            [xi, yi]    = geographicToIntrinsic(Rin, latq, lonq);
        elseif ~isGeographicRef(Rbase) && ~isGeographicRef(Rin)
            [xq,yq] = intrinsicToWorld(Rbase, Xc, Yc);
            [xi, yi]= worldToIntrinsic(Rin, xq, yq);
        else
            error('CRS mismatch base/input.');
        end
        if max(abs(xi(:)-(Xc(:)))./max(1,abs(Xc(:))))<1e-9 && ...
                max(abs(yi(:)-(Yc(:)))./max(1,abs(Yc(:))))<1e-9
            Aout = Ain; return;
        end
    catch
    end
end
[Xc,Yc] = meshgrid(1:Rbase.RasterSize(2), 1:Rbase.RasterSize(1));
try
    if isGeographicRef(Rbase) && isGeographicRef(Rin)
        [latq,lonq] = intrinsicToGeographic(Rbase, Xc, Yc);
        [xi, yi]    = geographicToIntrinsic(Rin, latq, lonq);
    elseif ~isGeographicRef(Rbase) && ~isGeographicRef(Rin)
        [xq,yq] = intrinsicToWorld(Rbase, Xc, Yc);
        [xi, yi]= worldToIntrinsic(Rin, xq, yq);
    else
        error('CRS mismatch.');
    end
catch ME
    error('Regrid "%s" failed: %s', name, ME.message);
end
try
    Aout = interp2(Ain, xi, yi, method, NaN);
catch
    warning('interp2 "%s" (%s) failed; using nearest.', name, method);
    Aout = interp2(Ain, xi, yi, 'nearest', NaN);
end
end

function [Rcorr, violCnt, violMag, repairedFrac, medAbsDelta] = enforceRatioMonotonicity(R, DURS, tol, clipHi)
if nargin<3 || isempty(tol),   tol = 1e-6;  end
if nargin<4 || isempty(clipHi), clipHi = 1.0; end
[nr,nc,nd] = size(R);
assert(nd==numel(DURS), 'Duration dimension mismatch.');
Rcorr    = R;
violCnt  = zeros(nr,nc,'single');
violMag  = zeros(nr,nc,'single');
Rcorr = max(Rcorr, 0);
Rcorr(:,:,end) = 1;
Rcorr(Rcorr>clipHi) = clipHi;
delta_list = [];
for p = 1:nr*nc
    [r,c] = ind2sub([nr,nc], p);
    rv = squeeze(Rcorr(r,c,:));
    if all(~isfinite(rv)), continue; end
    last = -Inf;  vc = 0;  vm = 0;  rv_fix = rv;
    for j = 1:nd
        x = rv_fix(j);
        if ~isfinite(x), continue; end
        m = max(last, x);
        if m > x + tol
            vc = vc + 1;
            vm = vm + (m - x);
            rv_fix(j) = m;
            delta_list(end+1) = abs(m - x); %#ok<AGROW>
        end
        last = rv_fix(j);
    end
    rv_fix(rv_fix>clipHi) = clipHi;
    rv_fix(end) = 1;
    Rcorr(r,c,:) = rv_fix;
    violCnt(r,c) = vc;
    violMag(r,c) = vm;
end
repairedFrac = nnz(violCnt)/numel(violCnt);
if isempty(delta_list), medAbsDelta = 0; else, medAbsDelta = median(delta_list); end
end

function [Rout, srcRow, srcCol, distPx] = fillRatiosFromNearestClean(Rin, isClean)
% Copy the entire ratio vector (all durations) for each problem pixel
% from its nearest clean pixel (Euclidean distance in row/col).
[nr,nc,nd] = size(Rin);
Rout = Rin;
problemMask = ~isClean;

if ~any(isClean(:)) || ~any(problemMask(:))
    srcRow = zeros(nr,nc,'single');
    srcCol = zeros(nr,nc,'single');
    distPx = inf(nr,nc,'single');
    return;
end

% Requires Image Processing Toolbox (bwdist). If missing, consider a KD-tree fallback.
[distPx, idxNearest] = bwdist(isClean, 'euclidean');

linProblem = find(problemMask);
srcLin     = idxNearest(linProblem);

[srcR, srcC] = ind2sub([nr,nc], srcLin);
[dstR, dstC] = ind2sub([nr,nc], linProblem);

for k = 1:numel(linProblem)
    Rout(dstR(k), dstC(k), :) = Rin(srcR(k), srcC(k), :);
end

srcRow = zeros(nr,nc,'single');  srcRow(linProblem) = single(srcR);
srcCol = zeros(nr,nc,'single');  srcCol(linProblem) = single(srcC);
distPx = single(distPx);
end

function [Dks, pval] = ks_gumbel_moments(x)
x = x(:); x = x(isfinite(x));
n = numel(x);
if n < 3, Dks = NaN; pval = NaN; return; end
m  = mean(x);
s  = std(x, 0);
beta = s / (pi/sqrt(6));                 % ≈ s / 1.28255
mu   = m - 0.5772156649015329 * beta;    % Euler-gamma * beta
xs = sort(x);
z  = (xs - mu) ./ beta;
F  = exp(-exp(-z));
i  = (1:n)'; Fe = (i - 0.5) / n;
Dks = max(abs(Fe - F));
en = sqrt(n);
lambda = (en + 0.12 + 0.11/en) * Dks;
pval = 0;
for j = 1:100
    pval = pval + (-1)^(j-1) * exp(-2*(j^2)*(lambda^2));
end
pval = max(min(2*pval,1),0);
end

function makeVRTWithBandNames(tifPath, bandNames, R, srsText)
info  = georasterinfo(tifPath);
nrows = info.RasterSize(1);
ncols = info.RasterSize(2);
[folder, base, ~] = fileparts(tifPath);
vrtPath = fullfile(folder, [base '_named.vrt']);
gt = geotransformFromRef(R);
dtype = 'Float32';
fid = fopen(vrtPath,'w');  assert(fid>0, 'Cannot write VRT: %s', vrtPath);
fprintf(fid,'<VRTDataset rasterXSize="%d" rasterYSize="%d">\n', ncols, nrows);
if nargin >= 4 && ~isempty(srsText)
    fprintf(fid,'  <SRS>%s</SRS>\n', srsText);
end
fprintf(fid,'  <GeoTransform>%.15g, %.15g, %.15g, %.15g, %.15g, %.15g</GeoTransform>\n', gt);
for b = 1:numel(bandNames)
    fprintf(fid,'  <VRTRasterBand dataType="%s" band="%d">\n', dtype, b);
    fprintf(fid,'    <Description>%s</Description>\n', bandNames{b});
    fprintf(fid,'    <SimpleSource>\n');
    fprintf(fid,'      <SourceFilename relativeToVRT="1">%s</SourceFilename>\n', [base '.tif']);
    fprintf(fid,'      <SourceBand>%d</SourceBand>\n', b);
    fprintf(fid,'      <SrcRect xOff="0" yOff="0" xSize="%d" ySize="%d"/>\n', ncols, nrows);
    fprintf(fid,'      <DstRect xOff="0" yOff="0" xSize="%d" ySize="%d"/>\n', ncols, nrows);
    fprintf(fid,'    </SimpleSource>\n');
    fprintf(fid,'  </VRTRasterBand>\n');
end
fprintf(fid,'</VRTDataset>\n');
fclose(fid);
fprintf('Wrote VRT with band names + CRS:\n  %s\n', vrtPath);
end

function gt = geotransformFromRef(R)
if isa(R,'map.rasterref.GeographicCellsReference')
    dx = R.CellExtentInLongitude;  dy = R.CellExtentInLatitude;
    xMin = R.LongitudeLimits(1);   yMax = R.LatitudeLimits(2);
    gt = [xMin, dx, 0, yMax, 0, -dy];
elseif isa(R,'map.rasterref.MapCellsReference')
    dx = R.CellExtentInWorldX;     dy = R.CellExtentInWorldY;
    xMin = R.XWorldLimits(1);      yMax = R.YWorldLimits(2);
    gt = [xMin, dx, 0, yMax, 0, -dy];
elseif isa(R,'map.rasterref.GeographicPostingsReference')
    dx = R.SampleSpacingInLongitude; dy = R.SampleSpacingInLatitude;
    xMin = R.LongitudeLimits(1) - dx/2;
    yMax = R.LatitudeLimits(2) + dy/2;
    gt = [xMin, dx, 0, yMax, 0, -dy];
elseif isa(R,'map.rasterref.MapPostingsReference')
    dx = R.SampleSpacingInWorldX;   dy = R.SampleSpacingInWorldY;
    xMin = R.XWorldLimits(1) - dx/2;
    yMax = R.YWorldLimits(2) + dy/2;
    gt = [xMin, dx, 0, yMax, 0, -dy];
else
    error('Unsupported raster reference class: %s', class(R));
end
end


function tf = isaRefGeographic(R)
tf = isa(R,'map.rasterref.GeographicCellsReference') || ...
    isa(R,'map.rasterref.GeographicPostingsReference');
end

function [latc, lonc] = cellCenterVectors(R)
if isaRefGeographic(R)
    dlat = R.CellExtentInLatitude;
    dlon = R.CellExtentInLongitude;
    latc = (R.LatitudeLimits(1)  + dlat/2) : dlat : (R.LatitudeLimits(2)  - dlat/2);
    lonc = (R.LongitudeLimits(1) + dlon/2) : dlon : (R.LongitudeLimits(2) - dlon/2);
    latc = latc(:); lonc = lonc(:).';
    latc = latc(1:R.RasterSize(1));
    lonc = lonc(1:R.RasterSize(2));
else
    error('cellCenterVectors: only implemented for geographic references.');
end
end

function [latq, lonq] = baseCellCenterGrid(Rbase)
[latv, lonv] = cellCenterVectors(Rbase);
[latq, lonq] = ndgrid(latv, lonv);
end


function [Bout, srcRow, srcCol, distPx] = fillFromNearestValid(Bin, isValid)
% Bin:  [nr x nc x nb] stack of bands to fill
% isValid: [nr x nc] logical mask of "good" pixels
% Fills every invalid pixel by copying the vector from the nearest valid pixel.
[nr,nc,nb] = size(Bin);
Bout   = Bin;
srcRow = zeros(nr,nc,'single');
srcCol = zeros(nr,nc,'single');

% If everything is valid or everything is invalid, handle trivially
if all(isValid(:))
    distPx = zeros(nr,nc,'single');
    return;
elseif ~any(isValid(:))
    % nothing to borrow; leave as-is
    distPx = inf(nr,nc,'single');
    return;
end

% Use image transform distance (requires Image Processing Toolbox)
[distPx, idxNearest] = bwdist(isValid, 'euclidean');

linBad   = find(~isValid);
srcLin   = idxNearest(linBad);

[srcR, srcC] = ind2sub([nr,nc], srcLin);
[dstR, dstC] = ind2sub([nr,nc], linBad);

for k = 1:numel(linBad)
    Bout(dstR(k), dstC(k), :) = Bin(srcR(k), srcC(k), :);
end

srcRow(linBad) = single(srcR);
srcCol(linBad) = single(srcC);
distPx         = single(distPx);
end
