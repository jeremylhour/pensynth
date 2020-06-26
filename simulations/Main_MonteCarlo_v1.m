% MODIFIED BY JEREMY ON 04 OCTOBER 2018

% Version 1: Only one r at a time

% Formula for r real
% bias correction implemented
% lambda minimizes MSE now
% RNG initialized for each worker - results are reproducible
% Pure synthetic control estimator implemented
% added x0=sqrt(x0).
% linear bias correction instead (see e-mail by Alberto on 25/09)
% reports nb. treated units outside convex hull

clear variables;
close all;
parpool('local',4);

cd '\\ulysse\users\JL.HOUR\1A_These\A. Research\RegSynthProject\regsynth\pensynth-matlab\output'

% Change resource parameters (walltime and mem) if necessary
%dcluster=parcluster;
%dcluster.ResourceTemplate='-l nodes=^N^,software=MATLAB_Distrib_Comp_Engine+^N^,walltime=24:00:00,mem=1000gb';
%dcluster.saveProfile;

%parpool('reducedform',100);
% Initialize RNG seed (specific to par-loop)
spmd
    rng(0,'combRecursive');
end

% Parameters

n1 = 10; % treated
n0 = 40; % control
k = 4; % dimension of covariates
a = 0.10; b = 0.90; h = 0.10; % support parameters
r = 1.2; % curvature of regression function

v = sqrt(k*((b^(2*r+1)-a^(2*r+1))/((b-a)*(2*r+1)) - (b^(r+1)-a^(r+1))^2/((b-a)*(r+1))^2)); % Normalizing constant

deltaLambda = 0.01;
firstLambda = 0.0001:deltaLambda:1;
maxLambda = 20;

%options = optimoptions('quadprog','StepTolerance',1e-10,'OptimalityTolerance',1e-10,'MaxIterations',2000,'Display','off');
options = optimoptions('quadprog','TolFun',1e-14,'TolX',1e-14,'MaxIter',2000,'Display','off');

T = 1000;
Estp = zeros(T,1); Estnp = zeros(T,1); Estm = zeros(T,1); Estmopt = zeros(T,1); Estpure = zeros(T,1);
MSEp = zeros(T,1); MSEnp = zeros(T,1); MSEm = zeros(T,1); MSEmopt = zeros(T,1); MSEpure = zeros(T,1);
Estp_bc = zeros(T,1); Estnp_bc = zeros(T,1); Estm_bc = zeros(T,1); Estmopt_bc = zeros(T,1); Estpure_bc = zeros(T,1);
MSEp_bc = zeros(T,1); MSEnp_bc = zeros(T,1); MSEm_bc = zeros(T,1); MSEmopt_bc = zeros(T,1); MSEpure_bc = zeros(T,1);
Densp = zeros(T,1); Densnp = zeros(T,1); Denspure = zeros(T,1); Denmopt =  zeros(T,1);
maxminDensp = zeros(T,2); maxminDensnp = zeros(T,2); maxminDenspure = zeros(T,2); maxminDenmopt = zeros(T,2);
OutConvHull = zeros(T,1);
lambdavalues = zeros(T,1);
M = 20; % Max number of matches for matching est.
dthr = 0.001; % threshold for a nul weight
margin = 25;
dlambda = margin*deltaLambda;
mvalues = zeros(T,1);

parfor t =1:T
    tic
    sprintf('Iteration: %d',t)
    
    stream = RandStream.getGlobalStream();
    stream.Substream = t;

    % 0. Simulate Data
    x1 = a+(b-a)*rand(k,n1);
    x0 = (a-h)+(b-a+2*h)*rand(k,n0);
    x0 = sqrt(x0);
    y1 = sum(x1.^r,1)'/v+randn(n1,1);
    y0 = sum(x0.^r,1)'/v+randn(n0,1);

    H = 2*(x0'*x0);
    
    minMSE = Inf;
    Wp = zeros(n0,n1);
    optlambda = 0;
    Lambda = firstLambda;

    % 1. Penalized Synthetic Control w/ optimized lambda
    % Here lambda is set to optimize MSE
    
    j = 1;
    while j <= length(Lambda)
        lambda = Lambda(j);
        W = zeros(n0,n1);
        for i=1:n1
            x = x1(:,i);
            D = x0 - kron(ones(1,n0),x);
            delta = diag(D'*D); 
            f = lambda*delta-2*x0'*x;
            w = quadprog(H,f,[],[],ones(1,n0),1,zeros(n0,1),ones(n0,1),[],options);
            W(:,i) = w;
        end
        mse = (y1-W'*y0)'*(y1-W'*y0);
        if mse < minMSE
            minMSE = mse;
            optlambda = lambda;
            Wp = W;
        end
        j = j + 1;
        if j > length(Lambda)         
             if   ((round(lambda-optlambda,5)<round(dlambda,5)) && (round(lambda+deltaLambda,5)<round(maxLambda,5)))
                Lambda = [Lambda lambda+(deltaLambda:deltaLambda:dlambda)];
            end
        end
    end
 
    lambdavalues(t) = optlambda;

    % 2. Matching
    
    % 2.1 Collects the indices of the m closet points in 'matches'
    matches = zeros(M,n1);
    for i=1:n1
         x = x1(:,i);
         D = x0 - kron(ones(1,n0),x);
         delta = diag(D'*D); 
         [sorted,I]=sort(delta);
         matches(:,i) = I(1:M);
    end
    
    % 2.2 Computes the corresponding MSE
    minMSEm = Inf;
    Wmopt = zeros(n0,n1);
    optm = 0;
    dim = size(Wmopt); 
    for m = 1:M
        W = zeros(n0,n1);
        W(sub2ind(dim,matches(1:m,:),ones(size(matches(1:m,:),1),1)*(1:dim(2))))=1/m;
        mse = (y1-W'*y0)'*(y1-W'*y0);
        if mse < minMSEm
            minMSEm = mse;
            optm = m;
            Wmopt = W;
        end
    end
    
    mvalues(t) = optm;

    % 3. Non-Penalized Synthetic Control
    Wnp = zeros(n0,n1);
    for i=1:n1
        x = x1(:,i);
        D = x0 - kron(ones(1,n0),x);
        delta = diag(D'*D); 
        w = quadprog(H,-2*x0'*x,[],[],ones(1,n0),1,zeros(n0,1),ones(n0,1),[],options);
        Wnp(:,i) = w;
    end
    
    % 4. "Pure" Synthetic Control (lambda close to zero case, see Th. 2)
    DT = delaunayn(x0'); 
    [tri,coordinates] = tsearchn(x0',DT,x1'); % return triangle and weight for each treated. NaN if outside conv. hull
    OutConvHull(t) = sum(isnan(tri));

    Wpure = zeros(n0,n1);
    for i = 1:n1
        if(isnan(tri(i))==0)
            idx = DT(tri(i),:);
            Wpure(idx,i) = coordinates(i,:);
        elseif(isnan(tri(i)))
            Wpure(:,i) = Wnp(:,i);
        end
    end
    
    % 5. One-to-One Matching
    Wm = zeros(n0,n1);
    Wm(sub2ind(dim,matches(1,:),ones(size(matches(1,:),1),1)*(1:dim(2))))=1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 6. Evaluate performance on subsequent period %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % simulate outcome for second period
    y1 = sum(x1.^r,1)'/v+randn(n1,1);
    y0 = sum(x0.^r,1)'/v+randn(n0,1);
    
    % Bias
    Estp(t) = mean(y1-Wp'*y0);
    Estnp(t) = mean(y1-Wnp'*y0);
    Estm(t) = mean(y1-Wm'*y0);
    Estmopt(t) = mean(y1-Wmopt'*y0);
    Estpure(t) = mean(y1-Wpure'*y0);

    % MSE
    MSEp(t) = (y1-Wp'*y0)'*(y1-Wp'*y0)/n1;
    MSEnp(t) = (y1-Wnp'*y0)'*(y1-Wnp'*y0)/n1;
    MSEm(t) = (y1-Wm'*y0)'*(y1-Wm'*y0)/n1;
    MSEmopt(t) = (y1-Wmopt'*y0)'*(y1-Wmopt'*y0)/n1; 
    MSEpure(t) = (y1-Wpure'*y0)'*(y1-Wpure'*y0)/n1;

    % Mean sparsity index 
    Densp(t) = mean(sum(Wp>dthr,1));
    Densnp(t) = mean(sum(Wnp>dthr,1));
    Denspure(t) = mean(sum(Wpure>dthr,1));
    Denmopt(t) = mean(sum(Wmopt>dthr,1));

    % Min and max of sparsity indices
    maxminDensp(t,:) = [min(sum(Wp>dthr)) max(sum(Wp>dthr))];
    maxminDensnp(t,:) = [min(sum(Wnp>dthr)) max(sum(Wnp>dthr))];
    maxminDenspure(t,:) = [min(sum(Wpure>dthr)) max(sum(Wpure>dthr))];
    maxminDenmopt(t,:) = [min(sum(Wmopt>dthr)) max(sum(Wmopt>dthr))];

    % 7. Bias correction (linear, quadratic commented)
    % f0 = [ones(1,n0); x0; x0.^2];
    % f1 = [ones(1,n1); x1; x1.^2];
    f0 = [ones(1,n0); x0];
    f1 = [ones(1,n1); x1];
    mu0 = (f0*f0')\(f0*y0);

    mu_hat0 = f0'*mu0;
    mu_hat1 = f1'*mu0;
    
    % bc: Bias
    Estp_bc(t) = mean(y1-Wp'*y0 - (mu_hat1-Wp'*mu_hat0));
    Estnp_bc(t) = mean(y1-Wnp'*y0 - (mu_hat1-Wnp'*mu_hat0));
    Estm_bc(t) = mean(y1-Wm'*y0 - (mu_hat1-Wm'*mu_hat0));
    Estmopt_bc(t) = mean(y1-Wmopt'*y0 - (mu_hat1-Wmopt'*mu_hat0));
    Estpure_bc(t) = mean(y1-Wpure'*y0 - (mu_hat1-Wpure'*mu_hat0));

    % bc: MSE
    MSEp_bc(t) = (y1-Wp'*y0 - (mu_hat1-Wp'*mu_hat0))'*(y1-Wp'*y0 - (mu_hat1-Wp'*mu_hat0))/n1;
    MSEnp_bc(t) = (y1-Wnp'*y0 - (mu_hat1-Wnp'*mu_hat0))'*(y1-Wnp'*y0 - (mu_hat1-Wnp'*mu_hat0))/n1;
    MSEm_bc(t) = (y1-Wm'*y0 - (mu_hat1-Wm'*mu_hat0))'*(y1-Wm'*y0 - (mu_hat1-Wm'*mu_hat0))/n1;
    MSEmopt_bc(t) = (y1-Wmopt'*y0 - (mu_hat1-Wmopt'*mu_hat0))'*(y1-Wmopt'*y0 - (mu_hat1-Wmopt'*mu_hat0))/n1; 
    MSEpure_bc(t) = (y1-Wpure'*y0 - (mu_hat1-Wpure'*mu_hat0))'*(y1-Wpure'*y0 - (mu_hat1-Wpure'*mu_hat0))/n1; 
    
    toc
end

% End of loop / saving results
filename = sprintf('n1_%d_n0_%d_k_%d_r_%d_%d%d_%d_T_%d_nested',n1,n0,k,r,100*a,100*b,100*h,T);
save(filename,'n1','n0','k','r','a','b','h','maxLambda','M','MSEp','MSEnp','MSEm','MSEmopt','Estp','Estnp','Estm','Estmopt','Densp','Densnp','Denmopt','MSEp_bc','MSEnp_bc','MSEm_bc','MSEmopt_bc','Estp_bc','Estnp_bc','Estm_bc','Estmopt_bc','maxminDensp','maxminDensnp','maxminDenmopt','lambdavalues','mvalues','T');

% Print to file and screen
Name = {'PenSynth';'NoPenSynth';'PureSynth';'Matching';'OptMatching';'PenSynth_bc';'NoPenSynth_bc';'PureSynth_bc';'Matching_bc';'OptMatching_bc'};
RMSEindiv = sqrt(mean([MSEp MSEnp MSEpure MSEm MSEmopt MSEp_bc MSEnp_bc MSEpure_bc MSEm_bc MSEmopt_bc]))';
RMSEatt = sqrt(mean([Estp Estnp Estpure Estm Estmopt Estp_bc Estnp_bc Estpure_bc Estm_bc Estmopt_bc].^2))';
Bias = abs(mean([Estp Estnp Estpure Estm Estmopt Estp_bc Estnp_bc Estpure_bc Estm_bc Estmopt_bc]))';
Sparsity = mean([Densp Densnp Denspure NaN(T,1) Denmopt NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1)])';
minSparsity = mean([maxminDensp(:,1) maxminDensnp(:,1) maxminDenspure(:,1) NaN(T,1) maxminDenmopt(:,1) NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1)])';
maxSparsity = mean([maxminDensp(:,2) maxminDensnp(:,2) maxminDenspure(:,2) NaN(T,1) maxminDenmopt(:,2) NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1)])';
Results = table(num2str(RMSEindiv,'%.4f'),num2str(RMSEatt,'%.4f'),num2str(Bias,'%.4f'),num2str(Sparsity,'%.4f'),num2str(minSparsity,'%.4f'),num2str(maxSparsity,'%.4f'),'RowNames',Name);
Results.Properties.VariableNames = {'RMSEindiv' 'RMSEatt' 'Bias' 'Sparsity' 'minSparsity' 'maxSparsity'}

txtname = sprintf('n1_%d_n0_%d_k_%d_r_%d_%d%d_%d_T_%d.txt',n1,n0,k,r,100*a,100*b,100*h,T);
writetable(Results,txtname,'Delimiter','\t','WriteRowNames',true);

OCHtxt = sprintf('Average nb. Treated outside Conv. Hull: %6.2f\n',mean(OutConvHull));
dlmwrite(txtname,OCHtxt,'-append','delimiter','','coffset',2);


delete(gcp('nocreate'));