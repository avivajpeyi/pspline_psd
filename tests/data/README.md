# DATASETS

* data0:
```R

library(psplinePsd)

set.seed(12345)

# Simulate AR(4) data
n = 2 ^ 7
ar.ex = c(0.9, -0.9, 0.9, -0.9)
data = arima.sim(n, model = list(ar = ar.ex))
data = data - mean(data)
data = data / stats::sd(data)

# Run MCMC (may take some time)
mcmc = gibbs_pspline(data, burnin=100, Ntotal=1000, degree = 3, eqSpacedKnots=TRUE)

# Plot result
plot(mcmc)

# Plot result on original scale with title
plot(mcmc, ylog = FALSE, main = "Estimate of PSD using the P-spline method")

save(list=c("data", "mcmc"), file="data.Rdata")


write.table(mcmc$tau, file="tau_trace.txt", row.names=FALSE, col.names=FALSE)
write.table(mcmc$ll.trace, file="ll_trace.txt", row.names=FALSE, col.names=FALSE)
write.table(mcmc$db.list, file="db_list.txt", row.names=FALSE, col.names=FALSE)

```
