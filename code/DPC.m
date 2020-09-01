function [dpmm, dpmm_posterior, dpmm_time] = DPC(X)
%% Dirichlet Process Mixture Model for Gaussian data
% 
% G ~ DP(a,H)
% H is a 2d Gaussian

%clear all; close all; 
rng('default');
options.generate_plots = 1;

%% DPMM parameters
if (nargin < 1)
    d=2;   %x in R^d
    n=1e3; %number of points
else
    [n,d] = size(X);
    ss = 1;
    sm = 3;
end

trueK=2;    %true number of clusters
trueZ=ceil(trueK*(1:n)/n); %true labels

K = 2;      %init number of clusters
alpha=1000;    %concentration parameter

num_gibbs = 1;

%% generate data
if (nargin < 1)
% synthetic data
ss = 1;
sm = 3;
mu = sm*randn(d,trueK);
X = transpose(mu(:,trueZ) + ss*randn(d,n)); %n x d
end

%% NIW prior parameters
%NIW(mu,S|m0,k0,nu0,S0)

niw.d  = d;
niw.mu0 = zeros(d,1);    %prior mean
niw.k0 = 1;              %how strongly we believe in m0 prior
niw.S0 = 2*ss^2*eye(d);  %prior mean for Sigma
niw.nu0 = d+1;  %3       %how strongly we believe in S0 prior (dof) nu0>D-1
niw.ss = sm^2/ss^2;      %cluster spread
niw.rr = 1/niw.ss;       %1/cluster spread              
niw.SS = niw.nu0*niw.S0; %nu0 * S0

%% DPMM algorithm

%pi=GEM(40,5);
%figure; stem(pi);

%init z (alternatively, can use k-means)
z = mod(randperm(n),K)+1;
%z = ceil(rand(1,n)*K);
%z = trueZ;

%init dpmm
fprintf('initializing dpmm...\n');
gauss = gauss_init(niw); %init gauss struct
[dpmm] = dpmm_init(K,alpha,gauss, X, z);

record_K = zeros(num_gibbs,1);
dpmm_time = zeros(num_gibbs,1);
Ns = [1 2 3 10 50 100 150 num_gibbs]; cnt = 1;
%Ns = 1:num_gibbs; cnt = 1;

for iter=1:num_gibbs
   
   record_K(iter) = dpmm.K;
   if (iter == Ns(cnt) && options.generate_plots == 1)
       cnt = cnt + 1;
       figure;
       dpmm_plot(dpmm);       
       axis([min(dpmm.X(:))-1 max(dpmm.X(:))+1  min(dpmm.X(:))-1 max(dpmm.X(:))+1]);
       title(['DPMM iter# ' num2str(iter)]);
       %saveas(gcf, strcat('./figures/dpmm_', num2str(iter), '.png'));
   end   
   fprintf('gibbs iter: %d\n', iter);
   
   if (iter == num_gibbs)
      numK = dpmm.K+1;
      dpmm_posterior = zeros(n,numK); 
   end
   
   tic;
   for ii = 1:n
       k = dpmm.z(ii);       
       %remove xi statistics from component zi       
       dpmm.nk(k) = dpmm.nk(k) - 1;       
       dpmm.gm{k} = del_item(dpmm.gm{k}, dpmm.X(ii,:));

       if (dpmm.nk(k) == 0)
           %delete empty component
           dpmm.gm(k) = [];
           dpmm.K = dpmm.K - 1;
           dpmm.nk(k) = [];          
           idx = find(dpmm.z>k);      
           dpmm.z(idx) = dpmm.z(idx) - 1;  %re-label component
       end
       
       log_pzi = log([dpmm.nk; dpmm.alpha]);  %p(zi=k|z_mi,alpha)       
       for kk=1:dpmm.K+1          
          %log p(zi=k|z_mi,x,alpha,beta) ~ log p(zi=k|z_mi,alpha) + log p(x_i|x_kmi, beta)
          log_pzi(kk) = log_pzi(kk) + log_predictive(dpmm.gm{kk}, dpmm.X(ii,:));
       end
       %sample zi ~ P(zi=k|z_mi,x,alpha,beta)
       pzi = exp(log_pzi - max(log_pzi)); %numerical stability
       pzi = pzi/sum(pzi);
       if (iter == num_gibbs), dpmm_posterior(ii,1:numK) = pzi(1:numK); end
       
       u = rand;
       k = find(u < cumsum(pzi),1,'first');
       
       if (k==dpmm.K+1)
           %create a new cluster
           dpmm.gm{k+1} = dpmm.gm{k}; %warm init
           dpmm.K = dpmm.K + 1;
           dpmm.nk= [dpmm.nk; 0];
       end
       
       %add xi statistics to component z_i=knew
       dpmm.z(ii) = k;
       dpmm.nk(k) = dpmm.nk(k) + 1;
       dpmm.gm{k} = add_item(dpmm.gm{k}, dpmm.X(ii,:));
       
   end 
   dpmm_time(iter) = toc;
end

SAVE_PATH = './';
save([SAVE_PATH,'dpmm.mat']);

%% generate plots
if (options.generate_plots)
figure; plot(record_K)
end

end

function [dpmm] = dpmm_init(K,alpha,gauss, X, z)

dpmm.K = K;
dpmm.N = size(X,1);
dpmm.alpha = alpha;
dpmm.gm = cell(1,K+1);  %cell of structs!
dpmm.X = X;
dpmm.z = z;
dpmm.nk = zeros(K,1);

%init mixture components
for kk = 1:K+1
    dpmm.gm{kk} = gauss;
end

%add data items to mixture components
for i = 1:dpmm.N    
    k = z(i);
    dpmm.gm{k} = add_item(dpmm.gm{k},X(i,:));
    dpmm.nk(k) = dpmm.nk(k) + 1;
end

end

function [gauss] = add_item(gauss, xi)

xi = xi(:);
gauss.n = gauss.n + 1;
gauss.rr = gauss.rr + 1;    
gauss.nu0 = gauss.nu0 + 1;
gauss.Sigma = cholupdate(gauss.Sigma,xi);
gauss.mu = gauss.mu + xi;

end

function [gauss] = del_item(gauss, xi)

xi = xi(:);
gauss.n = gauss.n - 1;
gauss.rr = gauss.rr - 1;    
gauss.nu0 = gauss.nu0 - 1;
gauss.Sigma = cholupdate(gauss.Sigma,xi,'-');
gauss.mu = gauss.mu - xi;

end

function [gauss] = gauss_init(niw)
%create a gaussian struct with no data items

gauss.d = niw.d;
gauss.n = 0;              % init number of items.
gauss.rr = niw.rr;
gauss.nu0 = niw.nu0;
gauss.Sigma = chol(niw.SS + niw.rr*niw.mu0*niw.mu0');
gauss.mu = niw.rr*niw.mu0;
gauss.Z0 = ZZ(niw.d,gauss.n,gauss.rr,gauss.nu0,gauss.Sigma,gauss.mu);

end

function ll = log_predictive(gauss,xi)
% log predictive probability of xx given other data items in the component
% log p(xi|x_1,...,x_n)
xi = xi(:);
ll =   ZZ(gauss.d,gauss.n+1,gauss.rr+1,gauss.nu0+1,cholupdate(gauss.Sigma,xi),gauss.mu+xi) ...
     - ZZ(gauss.d,gauss.n  ,gauss.rr  ,gauss.nu0  ,           gauss.Sigma    ,gauss.mu);
end

function zz = ZZ(dd,nn,rr,vv,CC,XX)

zz = - nn*dd/2*log(pi) ...
     - dd/2*log(rr) ...
     - vv*sum(log(diag( cholupdate(CC,XX/sqrt(rr),'-')  ))) ...
     + sum(gammaln((vv-(0:dd-1))/2));

end

function y = logdet(A)
% Compute log(det(A)) where A is positive-definite
% This is faster and more stable than using log(det(A)).
try
    U = chol(A);
    y = 2*sum(log(diag(U)));
catch
    y = 0;
    warning('logdet:posdef', 'Matrix is not positive definite');
end

end

function [pi_pdf]=GEM(K,alpha)
%stick-breaking construction
beta=zeros(K,1);
pi_pdf=zeros(K,1);

for k=1:K
    beta(k)=betarnd(1,alpha);
    pi_pdf(k)=beta(k)*prod(1-beta);
end

figure;
x=linspace(0,1,1e2);
plot(x,betapdf(x,1,5),'-b'); hold on;
plot(x,betapdf(x,1,10),'-r');
legend('\alpha=5','\alpha=10');
ylabel('beta(1,\alpha) pdf'); title('beta prior');

end

function [] = dpmm_plot(dpmm)

cmap = hsv(dpmm.K);

for kk = 1:dpmm.K
  [mu sigma] = rand_sample(dpmm.gm{kk});
  plotellipse(mu,sigma,'color',cmap(kk,:),'linewidth',2);
  hold on
  ii = find(dpmm.z==kk);
  d=length(mu);
  if (d > 2)
    [x_coeff, x_score] = pca(dpmm.X);
    x_pca = x_score(:,1:2);
    xx = x_pca(ii,:);
    plot(xx(:,1),xx(:,2),'.','color',cmap(kk,:),'markersize',10); hold on;
    xlabel('PC 1'); ylabel('PC 2'); title('PCA Plot');       
  else
    xx = dpmm.X(ii,:);
    plot(xx(:,1),xx(:,2),'.','color',cmap(kk,:),'markersize',10); hold on;
    xlabel('X1'); ylabel('X2');
  end
end

end

function el = plotellipse(xx,var,varargin)
% plot ellipse with centre xx and shape given by positive definite matrix var
[v,d] = eig(var);
d = sqrt(d);
theta = (0:.05:1)*2*pi;
xy = repmat(xx,1,length(theta))+d(1,1)*v(:,1)*sin(theta)+d(2,2)*v(:,2)*cos(theta);
dim = size(v,1);
if (dim <= 2)    
    el = plot(xy(1,:),xy(2,:),varargin{:});
end
end

function [mu,sigma] = rand_sample(gauss)
% [mu, sigma] = rand(qq)
% generates random mean mu and covariance sigma from the posterior of
% component parameters given data items in component.

CC = cholupdate(gauss.Sigma,gauss.mu/sqrt(gauss.rr),'-')\eye(gauss.d);

sigma = iwishrnd(CC,gauss.nu0,CC);
mu = mvnrnd(gauss.mu'/gauss.rr, sigma/gauss.rr)';

end

