function alpha = Sample_alpha(alpha_shape, alpha_rate, w, N)

rate = alpha_rate-sum(log(w),1);
scale = 1./rate;
alpha = gamrnd(N+alpha_shape, scale);