"""
ANALYSIS OF THE PREDICTED TEST OR VALIDATION RESULTS FROM RunMLP.py

@author:Mirudhula Mukundan
@date:11/29/2022
@ID:mirudhum
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


# get predictions
if os.path.getsize('predictions_b.pkl') > 0:
    with open('predictions_b.pkl','rb') as file:
        [p1,g1,IDs] = pickle.load(file)
else:
    print("File empty!")

pred = p1.squeeze()
gnd = g1.squeeze()

# get mean of all predicted cell age according to patient ID
patient_pred = []
patient_gnd = []
for i in np.unique(IDs):
    idx = np.where(IDs == i)
    p_idx = pred[idx]
    p_mean = np.mean(p_idx)
    patient_pred.append(round(p_mean))
    g_idx = gnd[idx]
    g_mean = np.mean(g_idx)
    patient_gnd.append(round(g_mean))

# MAE
sum1 = 0
for i in range(len(patient_gnd)):
    sum1 += abs(patient_gnd[i] - patient_pred[i])
  
mae_error = sum1/len(patient_gnd)

# Pearson Correlation
pearR = pearsonr(patient_pred,patient_gnd)

# R-squared value
reg = LinearRegression().fit(np.array(patient_gnd).reshape(-1,1), patient_pred)
R2 = reg.score(np.array(patient_gnd).reshape(-1,1), patient_pred)

# Linear regression line and scatterplot 
plt.plot(patient_gnd, m*np.array(patient_gnd)+b,color='black')
plt.plot(patient_gnd,patient_pred,'o',color='cornflowerblue')
plt.plot([0, 100], [0, 100], color = 'silver', linestyle='--')
plt.grid()
limitx1 = 35
limitx2 = 90
limity1 = 50
limity2 = 90
plt.xlim([limitx1, limitx2])
plt.ylim([limity1, limity2])
plt.title("Baseline 2 predictions on Validation \n(aggregate per patient)",fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Predicted Age")
R2_text = "R = "+str(round(R2,2))
plt.text(limitx1+5,limity2-7,R2_text,fontsize=14)
p_text = "p = "+str(round(pearR.pvalue,2))
plt.text(limitx1+5,limity2-10,p_text,fontsize=14)
mae_text = "MAE = "+str(round(mae_error,2))
plt.text(limitx2-15,limity1+5,mae_text,fontsize=14)


