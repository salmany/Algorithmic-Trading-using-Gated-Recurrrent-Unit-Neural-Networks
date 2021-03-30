# Algorithmic-Trading-using-Gated-Recurrrent-Unit-Neural-Networks

The aim of this study is to explore the viability of an autonomous algorithmic trading system employing advanced Artificial Intelligence methods that can exploit possible market inefficiencies in order to generate higher returns at the same (or lower) risk than the market returns. To supplement this research, a Robust Mean Variance Optimization model (based on Ledoit-Wolf, 2003) employing Resampling and Shrinkage techniques has been integrated, and considerable insights have been provided on the effectiveness of these independent traditional techniques for portfolio optimization. A python-based implementation has been used to develop a trading algorithm which works on the Robust Mean Variance Optimization model and provides optimized portfolios with and without Resampling and Shrinkage techniques. After application of the working prototype on actual data from NASDAQ on eight stocks, the optimized portfolio produced by the model has returned positive returns on actual data. These positive returns have been upheld with all three currently developed approaches: simple MVO optimization, Shrinkage on the covariance matrix as well as the Resampled optimized portfolio. The Resampled Mean Variance Optimization has been corroborated as the most efficient portfolio optimization technique based on practical runs. For a Deep Learning Algorithmic Trader, a Gated Recurrent Unit (GRU) based Recurrent Neural Network has been developed using TensorFlow on Python. Evaluations carried out on New York Stock Exchange trading data for a universe of eight stocks proves the effectiveness of A.I in predicting stock patterns, and the GRU / RNN Algorithmic Trader outperforms the Mean Variance Optimization model in terms of actualized portfolio returns.

Actual daily close returns will be used as training data, and optimized portfolios will be prepared using machine learning concepts as well as exclusively the robust mean variance optimized model. Portfolios will be rebalanced as deemed appropriate by all techniques, and gradually data will be developed on the different optimization techniques for comparison based on actualized returns.

METHODOLOGY

There are two practical methodologies for analyzing financial securities and subsequently making investment decisions. Incorporation of the right analysis methodology, or combination of methodologies, is of crucial importance for our model. Therefore, it requires meticulous planning and analysis, and an evaluation of the following two basic schools of thought in order to put together the most appropriate methodology.

1. Technical Analysis involves the use of a purely quantitative, statistical analysis of market activity. It does not focus on a security’s intrinsic value but rather exclusively on historical price and volume data.
There are three underlying assumptions that Technical Analysis works with:
•	Market discounts everything
•	Price changes occur in trends
•	History is of consequence to the future
There are various forms of technical analysis – from statistical indicators to graphical charts and patterns. This technique is often criticized for its exclusive consideration of pricing factors. However, the justification presented for this dependence is based on the Efficient Market Hypothesis, which basically argues that a security’s market price already incorporates all its possible influencing factors. This includes effects of market psychology, fundamental factors and any other perspectives which may influence the value of a security. Thus, the need for an independent analysis of these factors is eliminated and the sole variable becomes the analysis of price movements. 

2. Fundamental analysis focuses on the qualitative features of an asset in order to evaluate its intrinsic value. Anything that might influence the security’s perceived value is incorporated into the analysis. Macroeconomic factors, such as the economy’s health and the industry’s outlook, as well as microeconomic factors, such as company management and financial outlook, are accounted for to reach a quantitative valuation.
The underlying objective behind this technique is to classify between fundamentally strong and weak companies and industries. The value, performance and health of a company is evaluated through the examination and study of certain numerical and economic indicators.


Mean Variance Optimization Model

The optimization portfolio technique best suited to our model requirements seems to be the Robust Mean Variance Optimization method. This decision is arrived at keeping in mind the congruency between data inputs for the MVO and the data outputs of our proposed algorithm. The MVO works with expected means and the covariances in between the universe of securities; and the output of our algorithm shall be the forecasted prices – serving as expected means, and the subsequent calculation of covariances is trivial. Being a widely acclaimed method of portfolio optimization with much research to approve its practicality further cements our decision. 
However, there has also been much criticism of the MVO and many have worked on improving the method. The Robust Mean Variance Optimization is one such method which incorporates two techniques – resampling and shrinkage - for improving the robustness of the original method. This culminates in a methodology which results in better diversification, robustness and convexity in the frontier.



Implementation - Python

Python Libraries
A number of libraries were used in the implementation of our model:

1. NumPy
It is regarded as the fundamental package for scientific computing in Python. It provides N-Dimensional array objects with sophisticated inherent functions. There are useful branches developed for functionalities such as for using linear algebra and random number capabilities. 
Our model stores all matrices for returns, covariances and weights in the form of NumPy arrays.

2. Pandas
A Python library specialized for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series.

3. Pandas Remote Data Access
This is a further library of Pandas, which extracts data from various Internet sources into a pandas DataFrame. The data source used in our implementation is from Yahoo, which provides daily closing rates for all NASDAQ listed stocks.

4. SKLearn – Covariance and LedoitWolf
Provides many classification, datamining, regression and clustering algorithms along with being easily compatible with NumPy.
The sklearn.covariance package provides tools for accurately estimating a population’s covariance matrix under various settings.
This package also provides a Ledoit-Wolf Shrinkage technique which enables a comprehensive implementation of their Shrinkage model.

5. MatplotLib
MatplotLib is a plotting library for Python and NumPy. It provides functionality to produce 2D Graphs and Plots using python scripts with data based on NumPy arrays. Our model uses it to output a visual representation of the Efficient Frontier returned by the Mean Variance Optimization.

6. CvxPy
CVXPY is a Python-embedded package using modeling language for convex optimization problems. It allows the expression of problems in a natural way that follows the math, rather than in the restrictive standard form required by solvers.
Solvers could have alternatively been used for achieving the same results, but this intuitive package was used for personal convenience. 


Inputs – Stock Data
The model begins with a system of extracting historical returns. The returns can either be randomized using a multivariate normal distribution, or can be downloaded using Pandas from Yahoo Finance. 
Two variables are used to set the number of assets and the number of observations respectively, and there is no limit to either of these values. Any number of assets as well as observations have been tested to provide proper working with all aspects of our current model.
In case of using randomized returns, the NumPy functionality of producing a multivariate normal distribution is utilized to return a matrix of returns with dimensions of the stipulated number of assets and their number of observations. 
For downloading actual stock returns off the NASDAQ exchange, the Pandas Remote Access Library is used to request stock closing information from Yahoo. For our current model, we are using a universe of eight stocks.
These stocks were chosen to produce a universe of diverse assets spanning different industries. Alternatively, any listed company can be chosen by simply changing the ticker names of the data requests, and the model currently supports the use of the whole asset universe of the actual NASDAQ.
The number of observations chosen can be decided by the number of days we wish to obtain data for. This too is a variable amount and can stretch to as far back as recorded information exists on the exchange. There is no constraint in our model of the historical period to be used. 
For our current analysis, it was decided to use the data from 15th January, 2017 to 15th January, 2018 inclusive. A year of observations were taken, which sums to be 252 reflecting the number of actual trading days in any year. The closing data of these chosen stocks is then stored in a NumPy array and annualized to reflect their returns over a yearly basis. Subsequently, mean return and covariance matrices are calculated.

Shrinkage
To implement shrinkage, the above calculated covariance matrix is passed to the Ledoit-Wolf shrinkage package of SkLearn. A target matrix is also passed as a parameter to the function, which in our case is a multivariate normal distribution exhibiting a mean of zero, and the same covariance as the data of stock returns acquired.  An appropriate shrinkage factor is computed by the Ledoit-Wolf package and applied, reflecting the least distance between the target and original covariance matrix. A shrunk covariance matrix is thus computed.

Optimization
Using CvxPy, the optimization is carried out following the previously explained Risk Aversion technique. The problem is formulated in the terms of a variable set of weights, a parameter Gamma for representing the risk aversion factor and the returns and risks of the calculated portfolios. Constraints are also passed as an input, and CvxPy solves for an optimum set of weights. These weights are continually printed and output for different levels of Gamma, and a graph is formulated using a range of Gamma values. 

Resampling
The resampling functionality is independently  programmed. We begin by setting a number of iterations, which can range to any number. For our current analysis, a number of 100 iterations is used. A multivariate normal distribution is again invoked but this time with the same mean and covariance as the original data, and returns and covariance are calculated from this sample for every iteration. As part of every iteration, the CvxPy optimization outlined above is again solved drawing returns from the multivariate normal distribution, and the optimized weights calculated at each run is appended to a matrix. The returns, variance and weights of each iteration are output to the user. 
Once all iterations have been completed, the matrix of weights produced is average to obtain the resampled weights of the optimized portfolio.

Actualized Returns
Once the weights from the simple, resampled and shrunk MVO are calculated, the Pandas package is again called to acquire data for the next year. Continuing with our existing implementation, the data from 15th January 2018 onwards was collected till 28th December of the same year. These returns are then annualized, and the mean returns of all assets are calculated. This allows us to test if our optimized portfolios would earn the same returns as predicted.

Algorithmic Trading – An Artificial Intelligence Approach

There are numerous machine learning approaches that can be taken for constructing an A.I based algorithmic trader. For the purposes of a trader which is based on chronological trading data and is thus reliant on technical analysis, a technique which works well with a time-series dataset seems most appropriate and effective. In this context, one of the most popular and adaptive methods is the Long-Short-Term-Memory (LSTM) architecture, based on the use of a Recurrent Neural Network (RNN). A further advancement is achieved by a modified LSTM known as the Gated Recurrent Unit (GRU) mechanism, using a ‘forget gate’ and fewer parameters. 
Recurrent Neural Networks (RNN) are a class of Artificial Neural Networks (ANN) that are designed to recognize patterns in sequences of data. This includes information of almost any form, such as text, handwriting, verbal statements or (as is true in our case) numerical time series data. These algorithms focus on time and series as arguments, combining them to form a temporal dimension. By incorporating the element of ‘memory’, a Recurrent Neural Networks are able to recognize a data’s sequential characteristics and understand patterns to predict the next likely scenario. 
The concept of memory is achieved through feedback loops which help process a sequence of data that informs and influences the final output, thereby allowing information to persist throughout the neural network. In plain terms, RNNs are modified Artificial Neural Networks employing this use of ’memory’ through feedback loops. 

Simple Artificial Neural Networks constitute interconnected information processing nodes which are inspired from the structure of neurons in the human brain. They are created with layers of artificial neurons (or network nodes) which can absorb inputs and push outputs along the other layers in the network. The edges contain weights which determine the output’s influence and determine the network’s final output. Often, information is passed in a single direction in these networks – from inputs towards outputs. Such networks are known as ‘feedforward’ ANNs. An RNN is differentiated in this sense by its ability to be layered in a way which can process information in two directions. This functionality of looping around information back into the network is known as Backpropagation Through Time (BPTT).


Long-Short-Term-Memory (LSTM) and Gated Recurrent Units (GRUs)
One complication which arises with a feedforward RNN is the ‘vanishing gradient problem’, leading to a loss of performance due to ineffective training. This is more prevalent in deeply layered networks, as using a gradient based learning system degenerates itself as the network gets more complex. However, using LSTM units this problem can be overcome by the categorization of data into short- and long-term memory cells. This allows the selective looping back of important data into the network.
A Gated Recurrent Unit is the addition of a gating mechanism to the LSTM methodology. The term ‘gate’ is used to refer to neural networks that regulate the flow of information flowing through sequence chains, and selectively feed data back into the network. These gates help regulate the flow of information in the network and allow the algorithm to recognize which data in a sequence is important.
Unlike the LSTM cell with its three main components (cell state, input gate and output gate), the GRU cell has only two gates – the ‘reset’ gate and the ‘update’ gate. 
The ‘update gate’ decides which information to retain and which to discard, while the ‘reset gate’ decides how much past information to forget. Therefore, with fewer tensor operations GRUs tend to be more efficient while training than LSTMs. However, it is debatable as to which technique is more effective since both are best suited to their own specific types of problems. 
