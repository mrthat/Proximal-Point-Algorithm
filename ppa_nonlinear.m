%      ."".    ."",
%      |  |   /  /
%      |  |  /  /
%      |  | /  /
%      |  |/  ;-._ 
%      }  ` _/  / ;
%      |  /` ) /  /
%      | /  /_/\_/\
%      |/  /      |
%      (  ' \ '-  |
%       \    `.  /
%        |      |
%        |      |
%         
%         Jamie's Algorithm Collection No.3
%
%============= Proximal Point Algorithm (07/15/2013)============
% 
%               min       f(x)
%            subject to   g(x) <=  0
%                          x  \in  X (x>=0)
%=============================================================== 
% Proximal Points Algorithm for solving convex problems.
% This is nonlinear version using Bregman function h.
% by Jamie Zemin Zhang 07/15/2013.
%===============================================================
%

clear all
close all
clc

% ------------------ Overall Paramters -------------------
% ---------- Input an image, or generate one -------------

% Image   = imread('cameraman.tif')                       ;
Image   = imread('circles.png')                           ;
Image   = double(imresize(Image,0.08))                    ;
[m1,n1] = size(Image)                                     ;
n       = m1*n1                                           ;
x       = Image(:)                                        ;

% -------------- Operator A and observation b--------------
% noise is optional
% m       = 200                                             ;
% A       = randn(m,n)                                      ;
% noise   = norm(A*x)*randn(m,1)                            ;
% b       = A*x + 0.01*noise                                ;

% generate Radon Transform Operator
theta   = 0:5:180                                        ;
N       = 2*ceil(norm(size(Image) - ...
                          floor((size(Image)-1)/2)-1))+3  ;
A       = zeros( length(theta)*N , length(x) )            ;
I       = zeros( size(Image) )                            ;

for i = 1:m1
    for j = 1:n1
        temp = I                                          ;
        temp(j,i) = 1                                     ;
        R = radon(temp,theta,N)                           ;
        A(:,n1*(i-1)+j) = R(:)                            ;
    end
end

R       = A*x                                             ;
noise   = norm(A*x,inf) * randn(size(R))                  ;
b       = R + 0.01*noise                                  ;

figure;
subplot(131);imagesc(radon(Image,theta,N))                ;
title('Radon transform')                                  ;
subplot(132);imagesc(reshape(R,[ N , length(theta)]))     ;
title('Radon transform of image')                         ;
subplot(133);imagesc(reshape(b,[ N , length(theta)]))     ;
title('Add noise')                                        ;

% -------------- total-variation operator -----------------
D1      = zeros(n)                                        ;
D2      = zeros(n)                                        ;
for i = 1:m1*n1-1
    if mod(i,m1) ~= 0
        D1(i,i  )    =  1                                 ;
        D1(i,i+1)    = -1                                 ; 
    end
    
    if i <= m1*n1-m1
        D2(i,i)      =  1                                 ;
        D2(i,i+m1-1) = -1                                 ;
    end
end

% ================== Function definition ==================
% 
% f       ---- objective function
% f_subg  ---- subgradient of f
% g       ---- constraint function
% g_subg  ---- subgradient of g

% option  ---- choice for Bregman function
% h       ---- Bregman function(option: h1 , h2)
%              h1: \sum_{i}(x_i+sigma)log(x_i+sigma)
% h_subg  ---- subgradient of h 
% h_conj  ---- conjugate function of h
% h_c_pl  ---- conjugate plus function of h, defined as:
%              h_c_pl   = sup_{p \ge 0} {<p,z> - h(p)} 
%                        = h_conj(max{u,h_subg(0)})     
% h_c_sg  ---- subgradient of h_conj                
%

% Lg1     ---- Lipschitz constant for g and w1
% Lg2     ---- Lipschitz constant for g and w2
% B_w1    ---- Bregman distance of w1
% B_w2    ---- Bregman distance of w2
%
% =========================================================
  sigma   = 0.001                                         ;
  rho     = 0.005*norm(b)^2                               ;
  epsilon = 0                                             ;
  option  = 1                                             ;
  
%   f       = @(x)norm(x,1)                                 ;
%   f_subg  = @(x)sign(x)                                   ;
%   Lf      = 1                                             ;
%   
  f       = @(x)norm(D1*x,1)+norm(D2*x,1)                 ;
  f_subg  = @(x)D1'*sign(D1*x) + D2'*sign(D2*x)           ;
  Lf      = 2*max(sum(abs(D1)+abs(D2),2))                 ;
  
  g       = @(x)norm(A*x-b)^2 - rho                       ;     
  g_subg  = @(x)2*(A')*A*x - 2*A'*b                       ;
%   Lg1     = 2*max(max(abs(A'*A)))                         ;
  Lg1     = 1000;
  
switch option
    case 1
    h      = @(x) 0.5*x'*x                                ; 
    h_subg = @(x) x                                       ;
    h_conj = @(x) 0.5*x'*x                                ;
    h_c_pl = @(x) 0.5*max(x,0)'*max(x,0)                  ;
    h_c_sg = @(x) x                                       ;
  
    case 2
    h      = @(x) sum(x.*log(x)-x)                        ;
    h_subg = @(x) log(x)                                  ;
    h_conj = @(x) sum(exp(x))                             ;
    h_c_pl = @(x) sum(exp(x))                             ;
    h_c_sg = @(x) exp(x)                                  ;
end

% =========================================================

% -------------- parameters for the iteration -------------
max_ite   = 25                                            ;
max_inner = 100                                           ;
% first step size and  step size sequence
c0       = 0.00001                                        ;  
c        = 0.00001*ones(max_ite,1)                        ;  
% first guess
% x_new   = x + 0.001*norm(x,inf) * randn(size(x))            ;
x_new   = rand(size(x))                                   ;
p_new   = 0.1                                             ;

pri_res = norm( x_new   - x )                             ;
dual_res= norm( p_new   - 0 )                             ;
pri_tol = 0.001*norm(x)                                   ;
dual_tol= 0.001                                           ; 

ite     = 0                                               ;
silent  = 0                                               ;

s1='%3s\t\t%10s\t\t%10s\t\t%10s\t\t%10s\n'                ;
s2='%3d\t\t%10.4f\t\t%10.4f\t\t%10.0f\t\t%10.4f\n'        ;

% =================== main iteration ======================
if ~silent
    fprintf(s1, 'iter', 'x_gap' ,...
        'dual_gap' , 'inner_ite','inner residual');
end

tic
while ( pri_res >= pri_tol || dual_res >= dual_tol ) 
    ite = ite+1;
    
    
    x_old = x_new;
    p_old = p_new;
    
    % ----- start x update -----
    % first guess
    
    theta_new  = rand              ;
    theta_old  = theta_new         ;
    xx_new     = x_new             ;
    xx_old     = xx_new            ;
            
    inner_ite = 0                  ;       
    L         = 1                  ;
    xx_res    = 10                 ;
    xx_tol    = 0.001              ;
    
    % inner iteration
    while ( xx_res >= xx_tol )
                
        % disp inner iterations
        inner_ite  = inner_ite+1;
        
%         if mod(inner_ite , 20) ==0
%             wh = wdisp(0,sprintf('inner iteration %5d',inner_ite));
%             wh = wdisp(wh); 
%         end
        
        theta_older = theta_old ;
        theta_old   = theta_new ;
        xx_older    = xx_old    ;
        xx_old      = xx_new    ;
                
        % update
        yy_new     = xx_old + theta_old*(1/theta_older-1)*...
                     (xx_old-xx_older)                    ;
                 
%         norm_y = norm(yy_new)
                 
        % vector for use         
        const2     = f_subg(yy_new) + p_old * ...
                   exp(c(ite)*g(yy_new)) * g_subg(yy_new) ;         
                 
        xx_new     = yy_new - const2/Lg1                  ;
        
%         norm_x = norm(xx_new)
                
        theta_new  = ( -1*theta_old^2 + sqrt(theta_old^4+4*theta_old^2) )/2;
                
        xx_res      = norm( xx_new - xx_old )             ;

        if inner_ite == max_inner
            break;
        end
        
    end
    
    % x update
    x_new   =   xx_new .* (xx_new>=0)                     ; 
    % dual update
    p_new   = p_old * exp( c(ite)*g(x_new) )              ; 
            
    pri_res = norm( x_new - x_old   )/norm(x_old)         ;
    dual_res= norm( p_new - p_old   )/norm(p_old)         ;
    
    fprintf(s2,ite, pri_res, dual_res , inner_ite , xx_res )   ;
    
                                      
    if ite == max_ite
        break;
    end
        
end

x_est = x_new  ; 
t1 = toc       ;

%% ================ compare to CVX =======================

tic
cvx_begin quiet
    variable x_cvx(n)                                     ;
    minimize( norm(D1*x_cvx,1)+norm(D2*x_cvx,1)  )        ;
    subject to
        norm( A*x_cvx - b) <= sqrt(rho)                   ;
        x_cvx              >= 0                           ;
%         norm(x_cvx,1)      <= B                           ;
cvx_end
t2 = toc;

error_est        =  norm(x_est-x,'fro')/norm(x,'fro')     ;
error_cvx        =  norm(x_cvx-x,'fro')/norm(x,'fro')     ;

%% Plot and compare the results

figure;
subplot(231);imagesc(Image);title('Original Image')       ;

subplot(232);imagesc(reshape(x_est,[m1,n1]))              ;
title(['Nonlinear PPA using Bregman: ',num2str(error_est)])   ;

subplot(233);imagesc(reshape(x_cvx,[m1,n1]))              ;
title(['CVX Rec error: '     ,num2str(error_cvx)])        ;                                      

subplot(234);plot(x)                                      ;
subplot(235);plot(x_est);title(['time :',num2str(t1)])    ;
subplot(236);plot(x_cvx);title(['time :',num2str(t2)])    ;

disp(['error for nonlinear PPA: ',num2str(error_est)])    ;
disp(['error for cvx     : ',num2str(error_cvx)])         ;



%% =============== compare to comirror ===================


