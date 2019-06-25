function ls = logsum(loga, logb)
if loga < logb
    ls = logb+log(1+exp(loga-logb));
else
    ls = loga+log(1+exp(logb-loga));
end
