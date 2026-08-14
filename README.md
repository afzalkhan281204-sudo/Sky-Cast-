Scratch NumPy LSTM Weather Forecasting & Uncertainty Estimation System
A production-grade, mathematically verified time-series weather forecasting system built entirely from scratch using only NumPy. The model ingests a 30-day sequence of multi-variate weather observations and predicts the next 7 days of maximum daily temperatures (
T
m
a
x
T 
max
​
 ), decomposing predictions into epistemic (model) and aleatoric (data) uncertainty estimates.

🌟 Key Accomplishments & Metrics
Pure NumPy Architecture:
Built a complete LSTM recurrent layer from scratch with 4 gate mechanisms (Input 
i
t
i 
t
​
 , Forget 
f
t
f 
t
​
 , Candidate Cell 
C
~
t
C
~
  
t
​
 , Output 
o
t
o 
t
​
 ).
Implemented exact analytical Backpropagation Through Time (BPTT) gradient pass.
Built a scratch Adam Optimizer with bias correction and gradient norm clipping.
Mathematical Rigor & Gradient Check:
Verified analytical BPTT gradients against finite-difference numerical gradients (grad_check.py).
Achieved max relative gradient error of 
6.319
×
10
−
7
6.319×10 
−7
  (
<
10
−
6
<10 
−6
 ), mathematically proving gradient accuracy.
Uncertainty Quantification (Epistemic + Aleatoric):
Aleatoric Uncertainty: Softplus parametric Gaussian variance output head (
σ
^
2
σ
^
  
2
 ) trained via Negative Log-Likelihood (NLL).
Epistemic Uncertainty: Monte Carlo (MC) Dropout (
M
=
100
M=100) combined with a Deep Ensemble of 
K
=
3
K=3 independently initialized Scratch LSTM models.
Validation Performance:
Ensemble Validation MAE: 
2.30
∘
C
2.30 
∘
 C
Ensemble Validation RMSE: 
2.86
∘
C
2.86 
∘
 C
95% Confidence Interval Coverage (PICP): 
97.8
%
97.8% (Empirically well-calibrated).
Interactive Dark Glassmorphism Web Dashboard:
Live web application hosted at http://localhost:8080.
Real-time 7-day forecast chart with shaded 95% confidence bounds, uncertainty decomposition breakdown, historical 30-day lookback plots, and an interactive Weather Scenario Simulator (What-If Sandbox).
