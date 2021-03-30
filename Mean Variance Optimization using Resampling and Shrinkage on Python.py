# Generate data for long only portfolio optimization.
import numpy as np
import pandas_datareader.data as web
import pandas as pd
########### ONLINE STOCKS DATA COLLECTOR
stock_names = ['AAPL', 'AMZN', 'MSFT', 'HMC', 'GOOG', 'TM', 'TSLA', 'UN']
 
on_data = web.DataReader(stock_names,data_source="yahoo",start='1/15/2017', end='1/18/2018')['Adj Close']
 
on_data.sort_index(inplace=True)
ret_online = on_data.pct_change()
print(ret_online.shape) 
print(ret_online) 
ret_vec = np.asmatrix(ret_online).T
ret_vec = ret_vec[:,1:]
print(ret_vec)


###########
np.random.seed(1)

n_assets = 8
n_obs = 239
return_vec = ret_vec * 252
Sigma = np.cov(ret_vec) *252
#return_vec = np.random.randn(n_assets, n_obs)
mu = np.mean(return_vec,axis=1)
mu = mu 
print("Mean Matrix: ")
print(1 +mu)
mu.resize(n_assets,1)
Sigma = Sigma
print("Covariance Matrix:")
print(Sigma)
#SHRINKAGE Ledoit-Wolf
from sklearn.covariance import LedoitWolf
np.random.seed(1)
X = np.random.multivariate_normal(mean=np.zeros((n_assets,), dtype=int),
                                  cov=Sigma,
                                  size=50)
cov = LedoitWolf().fit(X)
print("Heres the covariance matrix:")
print(Sigma)
print("After Shrinkage:")
print(cov.covariance_) 
print("The optimal shrinkage factor was found to be: ", cov.shrinkage_)
Sigma = cov.covariance_
# Long only portfolio optimization.
from cvxpy import *
w = Variable(n_assets)
gamma = Parameter(nonneg=True)
ret = mu.T*w 
risk = quad_form(w, Sigma)
prob = Problem(Maximize(ret - gamma*risk), 
               [sum(w) == 1, 
                w >= 0])
gam = 3.05
gamma.value = gam
prob.solve()
print("Printing weights with Risk Aversion Factor: ", gamma.value)
print(np.round(w.value, decimals=2))
print("The corresponding returns are:" , ret.value, " with  variation: ", risk.value)
opt_w = w.value
# Compute trade-off curve.
plot = True
if plot == True:
		SAMPLES = 100
		risk_data = np.zeros(SAMPLES)
		ret_data = np.zeros(SAMPLES)
		gamma_vals = np.logspace(-2, 3, num=SAMPLES)
		for i in range(SAMPLES):
		    gamma.value = gamma_vals[i]
		    prob.solve()
		    risk_data[i] = sqrt(risk).value
		    ret_data[i] = ret.value



		# Plot long only trade-off curve.
		import matplotlib.pyplot as plt
		#%matplotlib inline
		#%config InlineBackend.figure_format = 'svg'

		markers_on = [29, 40]
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.plot(risk_data, ret_data, 'g-')
		for marker in markers_on:
		    plt.plot(risk_data[marker], ret_data[marker], 'bs')
		    ax.annotate(r"$\gamma = %.2f$" % gamma_vals[marker], xy=(risk_data[marker]+.08, ret_data[marker]-.03))
		for i in range(n_assets):
		    plt.plot(sqrt(Sigma[i,i]).value, mu[i], 'ro')
		plt.xlabel('Standard deviation')
		plt.ylabel('Return')
		plt.show()

resampling = True
if resampling == True:
	mu1 = mu
	print(mu1.T[0])
	print(mu1.T.tolist()[0])
	i = 100
	print(np.random.multivariate_normal(mean=mu1.T.tolist()[0],
	                                  cov=Sigma, size =1))
	prob.solve()
	print("Starting Iterations for Resampling")
	resampled = w.value
	for x in range(i):
		print("-----------------------------------")
		print("Initiating Iteration # ", x)
		y = np.random.multivariate_normal(mean=mu1.T.tolist()[0],cov=Sigma, size =1)
		b = 0
		j = y.tolist()[0]
		print(j)
		#print(sum(1 for number in j if j < 0))
		for x in j:
			if x < 0:
				print("WOW")
				b = b+1
		if b == len(j):
			print("BREAK")
			continue
		print(np.asmatrix(y))
		y = np.asmatrix(y[0])
		#print(y.T.shape, " -- ", mu.shape)
		w = Variable(n_assets)
		gamma = Parameter(nonneg=True)
		ret = y*w 
		Sigma = np.cov(ret_vec) 
		risk = quad_form(w, Sigma)

		prob = Problem(Maximize(ret - gamma*risk), 
		               [sum(w) == 1, ret >= 0,
	    	            w >= 0])
		gamma.value = gam
		prob.solve()
		print("Printing weights with Risk Aversion Factor: ", gamma.value)
		print(np.round(w.value, decimals=2))
		print("The corresponding returns are:" , ret.value, " with  variation: ", risk.value)
		resampled = np.append(np.asmatrix(resampled),np.asmatrix(w.value), axis = 0)
print("---------- RESAMPLING ITERATIONS COMPLETE ---------")
print("Resampled Weights Matrix:")
print(np.round(resampled, decimals = 2))
print("Mean of the resampled weights:")
print(np.round(np.mean(resampled, axis=0),decimals=2))
resampled = np.mean(resampled, axis=0)
print("Returns of the Resampled Portfolio:")
print(mu.T * np.mean(resampled, axis=0).T)
print("Variance of the Resampled Portfolio:")
print(np.sqrt(resampled * Sigma * resampled.T))


print("Fetching Actual Data for the next year:")
on_data = web.DataReader(stock_names,data_source="yahoo",start='1/18/2018', end='12/29/2018')['Adj Close']
 
on_data.sort_index(inplace=True)
ret_online = on_data.pct_change()

print(ret_online.shape) 
print(ret_online) 
ret_vec = np.asmatrix(ret_online).T
ret_vec = ret_vec[:,1:]
print(ret_vec)
return_vec = ret_vec * 252
Sigma = np.cov(return_vec) *252
#return_vec = np.random.randn(n_assets, n_obs)
mu = np.mean(return_vec,axis=1)
mu = mu 
print("Mean Matrix: ")
print(1 +mu)




print('Actual Returns with the Resampled portfolio:')
print(np.mean(resampled, axis=0) * mu)

print('Actual Returns with the MVO-only portfolio:')
print(opt_w*mu)
