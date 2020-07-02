% MODIFIED BY JAUME ON 30 JUNE 2020

% Version 2.5: based on 2.0 but adapted to compute lambdas/M for the bias
% corrected estimators.
% Probably not convenient for n0 = 1000

% Formula for r real
% bias correction implemented
% lambda minimizes MSE now
% RNG initialized for each worker - results are reproducible
% Pure synthetic control estimator implemented
% added x0=sqrt(x0).
% linear bias correction instead (see e-mail by Alberto on 25/09)
% reports nb. treated units outside convex hull
% several values of r at once

delete(gcp('nocreate'));
clear variables;
close all;
parpool('local',5);

%cd '\\ulysse\users\JL.HOUR\1A_These\A. Research\RegSynthProject...
%\regsynth\pensynth-matlab\output'

cd '/Users/jaumevives/Desktop/MIT 1/Abadie RAship/Output'

% Change resource parameters (walltime and mem) if necessary
%dcluster=parcluster;
%dcluster.ResourceTemplate='-l nodes=^N^,software=...
%MATLAB_Distrib_Comp_Engine+^N^,walltime=24:00:00,mem=1000gb';
%dcluster.saveProfile;

%parpool('reducedform',100);
% Initialize RNG seed (specific to par-loop)
spmd
    rng(0,'combRecursive');
end

% Number of iterations
T = 500;

% Parameters
n1 = 10; % treated
n0 = 20; % control
k = 4; % dimension of covariates
a = 0.1; b = 0.9; h = 0.1; % support parameters
rset = [1 1.2 1.4 2]; % curvature of regression function
nbr = size(rset,2);
dthr = 0.001; % threshold for a null weight

% Lambda grid
deltaLambda = 0.01;
firstLambda = 0.0001:deltaLambda:1;
maxLambda = 20;

lambdavalues = zeros(T,nbr);
lambdavaluesbc = zeros(T,nbr);

margin = 25;
dlambda = margin*deltaLambda;

% Mathing parameters
M = 20; % Max number of matches for matching est.
mvalues = zeros(T,nbr);
mvaluesbc = zeros(T,nbr);

% Quadprog options
%options = optimoptions('quadprog','StepTolerance',1e-10, ...
% 'OptimalityTolerance',1e-10,'MaxIterations',2000,'Display','off');
options = optimoptions('quadprog','TolFun',1e-14,'TolX', ...
                 1e-14,'MaxIter',2000,'Display','off');

% Initialize Statistics
% Bias
Estp = zeros(T,nbr); Estnp = zeros(T,nbr); Estm = zeros(T,nbr);
Estmopt = zeros(T,nbr); Estpure = zeros(T,nbr);

% MSE
MSEp = zeros(T,nbr); MSEnp = zeros(T,nbr); MSEm = zeros(T,nbr);
MSEmopt = zeros(T,nbr); MSEpure = zeros(T,nbr);

% Bias BC
Estp_bc = zeros(T,nbr); Estnp_bc = zeros(T,nbr); Estm_bc = zeros(T,nbr);
Estmopt_bc = zeros(T,nbr); Estpure_bc = zeros(T,nbr);

% MSE BC
MSEp_bc = zeros(T,nbr); MSEnp_bc = zeros(T,nbr); MSEm_bc = zeros(T,nbr);
MSEmopt_bc = zeros(T,nbr); MSEpure_bc = zeros(T,nbr);

% Density
Densp = zeros(T,nbr); Densnp = zeros(T,1); Denspure = zeros(T,1);
Denmopt =  zeros(T,nbr);
maxminDensp = zeros(T,2,nbr); maxminDensnp = zeros(T,2);
maxminDenspure = zeros(T,2); maxminDenmopt = zeros(T,2,nbr);
OutConvHull = zeros(T,1); OutConvHull_M2 = zeros(T,1);

% Parfor loop
parfor t = 1:T
    tic
    sprintf('Iteration: %d',t)
    
    stream = RandStream.getGlobalStream();
    stream.Substream = t;

    % 0. Simulate Data
    % Notice: x1 and x0 are simulated only once per loop
    x1 = a+(b-a)*rand(k,n1);
    x0 = (a-h)+(b-a+2*h)*rand(k,n0);
    x0 = sqrt(x0);
    
    H = 2*(x0'*x0);
    f0 = [ones(1,n0); x0];
    f1 = [ones(1,n1); x1];
    
    eps1 = randn(n1,1); eps0 = randn(n0,1);
    y1 = NaN(n1,nbr); y0 = NaN(n0,nbr); 
    i=0;
    
    for r = rset
        i = i+1;
        v = sqrt(k*((b^(2*r+1)-a^(2*r+1))/((b-a)*(2*r+1)) -...
            (b^(r+1)-a^(r+1))^2/((b-a)*(r+1))^2)); % Normalizing constant
        y1(:,i) = sum(x1.^r,1)'/v+eps1;
        y0(:,i) = sum(x0.^r,1)'/v+eps0;
    end


    
    
    % 1. Penalized Synthetic Control w/ optimized lambda
    % Here lambda is set to optimize the MSE and the BC MSE
    
    % Initialize penalized synthetic control
    Wp = NaN(n0,n1,nbr);
    optlambdacollect = [];
    
    % Initialize penalized synthetic control with bias-correction
    Wpbc = NaN(n0,n1,nbr);
    optlambdacollectbc = [];
    
    ii = 0;
    for r = rset
        ii=ii+1;
        
        Lambda = firstLambda;
        
        minMSE = Inf;
        optlambda = 0;
        
        % compute OLS estimator for the bias-correction
        mu0 = (f0*f0')\(f0*y0(:,ii));
        mu_hat0 = f0'*mu0;
        mu_hat1 = f1'*mu0;
        
        minMSEbc = Inf;
        optlambdabc = 0;
        
        % Iterate over lambda grid
        j = 1;
        while j <= length(Lambda)
            lambda = Lambda(j);
            W = zeros(n0,n1);
            
            % Solve for w
            for z=1:n1
                x = x1(:,z);
                D = x0 - kron(ones(1,n0),x);
                delta = diag(D'*D); 
                f = lambda*delta-2*x0'*x;
                w = quadprog(H,f,[],[],ones(1,n0),1,zeros(n0,1),...
                    ones(n0,1),[],options);
                W(:,z) = w;
            end
            
            % Compute MSEs
            mse = (y1(:,ii)-W'*y0(:,ii))'*(y1(:,ii)-W'*y0(:,ii));
            msebc = (y1(:,ii)-W'*y0(:,ii)...
                - (mu_hat1-W'*mu_hat0))'*(y1(:,ii)-W'*y0(:,ii)...
                - (mu_hat1-W'*mu_hat0));
            
            % Update min lambda
            if mse < minMSE
                minMSE = mse;
                optlambda = lambda;
                Wp(:,:,ii) = W;
            end
            
            if msebc < minMSEbc
                minMSEbc = msebc;
                optlambdabc = lambda;
                Wpbc(:,:,ii) = W;
            end
            
            % Increase grid size if last lambda optimal
            j = j + 1;
            if j > length(Lambda)         
                if (((round(lambda-optlambda,5)<round(dlambda,5))...
                      && (round(lambda+deltaLambda,5)<round(maxLambda,5))))...
                    || (((round(lambda-optlambdabc,5)<round(dlambda,5))...
                      && (round(lambda+deltaLambda,5)<round(maxLambda,5))))
                    
                  Lambda = [Lambda...
                        lambda+(deltaLambda:deltaLambda:dlambda)];
                end
            end
        end
        optlambdacollect = [optlambdacollect; optlambda];
        optlambdacollectbc = [optlambdacollectbc; optlambdabc];
    end
    
    % Save optimal lambdas
    lambdavalues(t,:) = optlambdacollect;
    lambdavaluesbc(t,:) = optlambdacollectbc;

    % 2. Nearest Neighbor Matching
    
    % 2.1 Collects the indices of the m closet points in 'matches'
    matches = zeros(M,n1);
    for z=1:n1
         x = x1(:,z);
         D = x0 - kron(ones(1,n0),x);
         delta = diag(D'*D); 
         [sorted,I]=sort(delta);
         matches(:,z) = I(1:M);
    end
    
    % 2.2 Computes the corresponding MSE
    % Here we compute the MSE and the bias-corrected MSE
    Wmopt = zeros(n0,n1,nbr);
    optmcollect = [];
    
    Wmoptbc = zeros(n0,n1,nbr);
    optmcollectbc = [];
    
    dim = size(Wmopt(:,:,1)); 
    
    i = 0;
    for r = rset
        i=i+1;
        
        % for the BC
        mu0 = (f0*f0')\(f0*y0(:,i));
        mu_hat0 = f0'*mu0;
        mu_hat1 = f1'*mu0;
        
        minMSEm = Inf;
        minMSEmbc = Inf;
        
        optm = 0;
        optmbc = 0;
        
        % Iterate over matching number
        for m = 1:M
            W = zeros(n0,n1);
            W(sub2ind(dim,matches(1:m,:), ...
                ones(size(matches(1:m,:),1),1)*(1:dim(2))))=1/m;
            
            % Compute MSEs
            mse = (y1(:,i)-W'*y0(:,i))'*(y1(:,i)-W'*y0(:,i));
            msebc = (y1(:,i)-W'*y0(:,i)...
                - (mu_hat1-W'*mu_hat0))'*(y1(:,i)-W'*y0(:,i)...
                - (mu_hat1-W'*mu_hat0));
            
            % Update min lambda
            if mse < minMSEm
                minMSEm = mse;
                optm = m;
                Wmopt(:,:,i) = W;
            end
            
            if msebc < minMSEmbc
                minMSEmbc = msebc;
                optmbc = m;
                Wmoptbc(:,:,i) = W;
            end
        end
        optmcollect = [optmcollect; optm];
        optmcollectbc = [optmcollectbc; optmbc];
    end
    
    % Save optimal m
    mvalues(t,:) = optmcollect;
    mvaluesbc(t,:) = optmcollectbc;
    
    
    % NB: only step 1 and 2 depend on y (and r) 
    % Non-penalized synthetic control, pure synthetic control and 1-NN-matching are computed without "y"

    % 3. Non-Penalized Synthetic Control (lambda=0)
    Wnp = zeros(n0,n1);
    for z=1:n1
        x = x1(:,z);
        D = x0 - kron(ones(1,n0),x);
        delta = diag(D'*D); 
        w = quadprog(H,-2*x0'*x,[],[],ones(1,n0),1,zeros(n0,1),ones(n0,1)...
            ,[],options);
        Wnp(:,z) = w;
    end
    
    % 3-5. Count Outside Convex Hull (Second method)
    OutConvHull_M2(t) = sum(diag((x1-x0*Wnp)'*(x1-x0*Wnp)) > 0.0001);
    
    % 4. "Pure" Synthetic Control (lambda close to zero case, see Th. 2)
    DT = delaunayn(x0'); 
    
    % return triangle and weight for each treated. NaN if outside conv.hull
    [tri,coordinates] = tsearchn(x0',DT,x1'); 
    OutConvHull(t) = sum(isnan(tri));

    Wpure = zeros(n0,n1);
    for z = 1:n1
        if(~isnan(tri(z)))
            idx = DT(tri(z),:);
            Wpure(idx,z) = coordinates(z,:);
        elseif(isnan(tri(z)))
            Wpure(:,z) = Wnp(:,z);
        end
    end
    
    % 5. One-to-One Matching
    Wm = zeros(n0,n1);
    Wm(sub2ind(dim,matches(1,:),...
        ones(size(matches(1,:),1),1)*(1:dim(2))))=1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 6. Evaluate performance on subsequent period %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % simulate outcome for second period
    eps1 = randn(n1,1); eps0 = randn(n0,1);
    y1 = NaN(n1,nbr); y0 = NaN(n0,nbr); 
    i = 0;
    
    % Initialize placeholders
    Estp_Temp = []; Estnp_Temp = []; Estm_Temp = [];
    Estmopt_Temp = []; Estpure_Temp = [];
    
    MSEp_Temp = []; MSEnp_Temp = []; MSEm_Temp = [];
    MSEmopt_Temp = []; MSEpure_Temp = [];
    
    Densp_Temp = []; Denmopt_Temp = [];
    maxminDensp_Temp = []; maxminDenmopt_Temp = [];
    
    % Iterate over rset and compute period 1 statistics
    for r = rset
        i = i+1;
        v = sqrt(k*((b^(2*r+1)-a^(2*r+1))/((b-a)*(2*r+1))...
            - (b^(r+1)-a^(r+1))^2/((b-a)*(r+1))^2)); % Normalizing constant
        y1(:,i) = sum(x1.^r,1)'/v+eps1;
        y0(:,i) = sum(x0.^r,1)'/v+eps0;
    
        % Bias
        Estp_Temp = [Estp_Temp; mean(y1(:,i)-Wp(:,:,i)'*y0(:,i))];
        Estnp_Temp = [Estnp_Temp; mean(y1(:,i)-Wnp'*y0(:,i))];
        Estm_Temp = [Estm_Temp; mean(y1(:,i)-Wm'*y0(:,i))];
        Estmopt_Temp = [Estmopt_Temp; mean(y1(:,i)-Wmopt(:,:,i)'*y0(:,i))];
        Estpure_Temp = [Estpure_Temp; mean(y1(:,i)-Wpure'*y0(:,i))];

        % MSE
        MSEp_Temp = [MSEp_Temp;...
            (y1(:,i)-Wp(:,:,i)'*y0(:,i))'*(y1(:,i)-Wp(:,:,i)'*y0(:,i))/n1];
        MSEnp_Temp = [MSEnp_Temp;...
            (y1(:,i)-Wnp'*y0(:,i))'*(y1(:,i)-Wnp'*y0(:,i))/n1];
        MSEm_Temp = [MSEm_Temp;...
            (y1(:,i)-Wm'*y0(:,i))'*(y1(:,i)-Wm'*y0(:,i))/n1];
        MSEmopt_Temp = [MSEmopt_Temp;...
            (y1(:,i)-Wmopt(:,:,i)'*y0(:,i))'*(y1(:,i)-Wmopt(:,:,i)'*y0(:,i))/n1]; 
        MSEpure_Temp = [MSEpure_Temp;...
            (y1(:,i)-Wpure'*y0(:,i))'*(y1(:,i)-Wpure'*y0(:,i))/n1];

        % Mean sparsity index 
        Densp_Temp = [Densp_Temp; mean(sum(Wp(:,:,i)>dthr,1))];
        Denmopt_Temp = [Denmopt_Temp; mean(sum(Wmopt(:,:,i)>dthr,1))];
        
        % Min and max of sparsity indices
        maxminDensp_Temp = [maxminDensp_Temp;
                        min(sum(Wp(:,:,i)>dthr)) max(sum(Wp(:,:,i)>dthr))];
        maxminDenmopt_Temp = [maxminDenmopt_Temp;
                  min(sum(Wmopt(:,:,i)>dthr)) max(sum(Wmopt(:,:,i)>dthr))];
    end
    
    Estp(t,:) = Estp_Temp; Estnp(t,:) = Estnp_Temp; Estm(t,:) = Estm_Temp;
    Estmopt(t,:) = Estmopt_Temp; Estpure(t,:) = Estpure_Temp;
    
    MSEp(t,:) = MSEp_Temp; MSEnp(t,:) = MSEnp_Temp; MSEm(t,:) = MSEm_Temp;
    MSEmopt(t,:) = MSEmopt_Temp; MSEpure(t,:) = MSEpure_Temp;
    
    Densp(t,:) = Densp_Temp; Denmopt(t,:) = Denmopt_Temp;
    
    maxminDensp(t,:,:) = maxminDensp_Temp';
    maxminDenmopt(t,:,:) = maxminDenmopt_Temp';
    
    % Sparsity stats for those who do not depend on r
    Densnp(t) = mean(sum(Wnp>dthr,1));
    Denspure(t) = mean(sum(Wpure>dthr,1));
    
    maxminDensnp(t,:) = [min(sum(Wnp>dthr)) max(sum(Wnp>dthr))];
    maxminDenspure(t,:) = [min(sum(Wpure>dthr)) max(sum(Wpure>dthr))];

    % 7. Bias correction
    i = 0;
    
    Estp_Temp = []; Estnp_Temp = []; Estm_Temp = [];
    Estmopt_Temp = []; Estpure_Temp = [];
    
    MSEp_Temp = []; MSEnp_Temp = []; MSEm_Temp = [];
    MSEmopt_Temp = []; MSEpure_Temp = [];
    
    for r = rset
        i=i+1;
        mu0 = (f0*f0')\(f0*y0(:,i));

        mu_hat0 = f0'*mu0;
        mu_hat1 = f1'*mu0;
    
        % bias-correction: Bias
        Estp_Temp = [Estp_Temp;...
            mean(y1(:,i)-Wpbc(:,:,i)'*y0(:,i)...
            - (mu_hat1-Wpbc(:,:,i)'*mu_hat0))];
        
        Estnp_Temp = [Estnp_Temp;...
            mean(y1(:,i)-Wnp'*y0(:,i)...
            - (mu_hat1-Wnp'*mu_hat0))];
        
        Estm_Temp = [Estm_Temp;...
            mean(y1(:,i)-Wm'*y0(:,i)...
            - (mu_hat1-Wm'*mu_hat0))];
        
        Estmopt_Temp = [Estmopt_Temp;...
            mean(y1(:,i)-Wmoptbc(:,:,i)'*y0(:,i)...
            - (mu_hat1-Wmoptbc(:,:,i)'*mu_hat0))];
        
        Estpure_Temp = [Estpure_Temp;...
            mean(y1(:,i)-Wpure'*y0(:,i)...
            - (mu_hat1-Wpure'*mu_hat0))];

        % bias correction: MSE
        MSEp_Temp = [MSEp_Temp;...
            (y1(:,i)-Wpbc(:,:,i)'*y0(:,i)...
            - (mu_hat1-Wpbc(:,:,i)'*mu_hat0))'*(y1(:,i)-Wpbc(:,:,i)'*y0(:,i)...
            - (mu_hat1-Wpbc(:,:,i)'*mu_hat0))/n1];
        
        MSEnp_Temp = [MSEnp_Temp;...
            (y1(:,i)-Wnp'*y0(:,i)...
            - (mu_hat1-Wnp'*mu_hat0))'*(y1(:,i)-Wnp'*y0(:,i)...
            - (mu_hat1-Wnp'*mu_hat0))/n1];
        
        MSEm_Temp = [MSEm_Temp;...
            (y1(:,i)-Wm'*y0(:,i)...
            - (mu_hat1-Wm'*mu_hat0))'*(y1(:,i)-Wm'*y0(:,i)...
            - (mu_hat1-Wm'*mu_hat0))/n1];
        
        MSEmopt_Temp = [MSEmopt_Temp;...
            (y1(:,i)-Wmoptbc(:,:,i)'*y0(:,i)...
            - (mu_hat1-Wmoptbc(:,:,i)'*mu_hat0))'*(y1(:,i)-Wmoptbc(:,:,i)'*y0(:,i)...
            - (mu_hat1-Wmoptbc(:,:,i)'*mu_hat0))/n1]; 
        
        MSEpure_Temp = [MSEpure_Temp;...
            (y1(:,i)-Wpure'*y0(:,i)...
            - (mu_hat1-Wpure'*mu_hat0))'*(y1(:,i)-Wpure'*y0(:,i)...
            - (mu_hat1-Wpure'*mu_hat0))/n1]; 
    end
    
    Estp_bc(t,:) = Estp_Temp; Estnp_bc(t,:) = Estnp_Temp;
    Estm_bc(t,:) = Estm_Temp; Estmopt_bc(t,:) = Estmopt_Temp;
    Estpure_bc(t,:) = Estpure_Temp;
    
    MSEp_bc(t,:) = MSEp_Temp; MSEnp_bc(t,:) = MSEnp_Temp;
    MSEm_bc(t,:) = MSEm_Temp; MSEmopt_bc(t,:) = MSEmopt_Temp;
    MSEpure_bc(t,:) = MSEpure_Temp;
    
    toc
end

% End of loop / saving results
filename = sprintf('n1_%d_n0_%d_k_%d_%d%d_%d_T_%d_nested',n1,n0,k,...
                    100*a,100*b,100*h,T);
save(filename,'n1','n0','k','rset','a','b','h','maxLambda','M',...
'MSEp','MSEnp','MSEm','MSEmopt','Estp','Estnp','Estm','Estmopt',...
'Densp','Densnp','Denmopt','MSEp_bc','MSEnp_bc','MSEm_bc','MSEmopt_bc',...
'Estp_bc','Estnp_bc','Estm_bc','Estmopt_bc','maxminDensp','maxminDensnp',...
'maxminDenmopt','lambdavalues','mvalues','T');

i = 0;
for r = rset
    i=i+1;
    % Print to file and screen
    Name = {'PenSynth';'NoPenSynth';'PureSynth';'Matching';'OptMatching';...
            'PenSynth_bc';'NoPenSynth_bc';'PureSynth_bc';'Matching_bc';...
            'OptMatching_bc'};
        
    RMSEindiv = sqrt(mean([MSEp(:,i) MSEnp(:,i) MSEpure(:,i) MSEm(:,i)...
        MSEmopt(:,i) MSEp_bc(:,i) MSEnp_bc(:,i) MSEpure_bc(:,i)...
        MSEm_bc(:,i) MSEmopt_bc(:,i)]))';
    
    RMSEatt = sqrt(mean([Estp(:,i) Estnp(:,i) Estpure(:,i) Estm(:,i)...
        Estmopt(:,i) Estp_bc(:,i) Estnp_bc(:,i) Estpure_bc(:,i)...
        Estm_bc(:,i) Estmopt_bc(:,i)].^2))';
    
    Bias = abs(mean([Estp(:,i) Estnp(:,i) Estpure(:,i) Estm(:,i)...
        Estmopt(:,i) Estp_bc(:,i) Estnp_bc(:,i) Estpure_bc(:,i)...
        Estm_bc(:,i) Estmopt_bc(:,i)]))';
    
    Sparsity = mean([Densp(:,i) Densnp Denspure NaN(T,1) Denmopt(:,i)...
        NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1)])';
    
    minSparsity = mean([maxminDensp(:,1,i) maxminDensnp(:,1)...
        maxminDenspure(:,1) NaN(T,1) maxminDenmopt(:,1,i)...
        NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1)])';
    
    maxSparsity = mean([maxminDensp(:,2,i) maxminDensnp(:,2)...
        maxminDenspure(:,2) NaN(T,1) maxminDenmopt(:,2,i)...
        NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1) NaN(T,1)])';
    
    Results = table(num2str(RMSEindiv,'%.4f'),num2str(RMSEatt,'%.4f'),...
        num2str(Bias,'%.4f'),num2str(Sparsity,'%.4f'),...
        num2str(minSparsity,'%.4f'),num2str(maxSparsity,'%.4f'),...
        'RowNames',Name);
    
    Results.Properties.VariableNames = {'RMSEindiv' 'RMSEatt' 'Bias'...
                                    'Sparsity' 'minSparsity' 'maxSparsity'};

    txtname = sprintf('n1_%d_n0_%d_k_%d_r_%d_%d%d_%d_T_%d.txt',n1,...
                       n0,k,r,100*a,100*b,100*h,T);
                   
    writetable(Results,txtname,'Delimiter','\t','WriteRowNames',true);

    OCHtxt = sprintf('Average nb. Treated outside Conv. Hull: %6.2f\n',...
                      mean(OutConvHull));
                  
    dlmwrite(txtname,OCHtxt,'-append','delimiter','','coffset',2);
end

delete(gcp('nocreate'));
