function alpha = Sample_alpha(w,N,alpha_shape,alpha_rate)

rate = alpha_rate-sum(log(w),1);
scale = 1./rate;
alpha = gamrnd(N+alpha_shape,scale);