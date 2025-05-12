#CCT Midterm
#Lily Dorval
#Cogs 107 
#Professor Joachim Vandekerckhove

#Report

#The model defines D as a variable representing each informants competence, the probability that they know the correct answer
#Each D[i] is drawn from a uniform prior between 0.5 and 1.0, where 0.5 is random chancea and 1.0 is perfect competencce.
#Z[j] is the latent consensus, or "true" answers for each question j.
#Each Z[j] is modeled as a Bernoulli trial with a 50% prior for being "true"
#A matrix p[i,j], the probability that informant i gives a "yes" response for item j is created.
#This encodes an assumption of CCT, that informants agree with consensus in proportion to their competence.
#X_obs is the likelihood that the matrix X (observed data) is modeled as Bernoulli with the probability matrix p.
# 2000 samples are drawn per chain after 1000 tuning steps.
#With relative consistency, informant D[5] is identified as the most competent and D[2] as the least competent.
#Note that these are indices, so these are informants 6 and 3, not 5 and 2.
#Convergence is always good, with uniform 1.0 r_hat for all Z[j] and D[i]
#The naive and CCT Item answers are similar but not identical.
#All proposed answers are the same in both models except for Questions 2, 13, 14 and 15.


import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# gets data with pandas and converts to numpy array
def getData():
    link = 'https://raw.githubusercontent.com/joachimvandekerckhove/cogs107s25/refs/heads/main/1-mpt/data/plant_knowledge.csv'
    df = pd.read_csv(link)

    #cleaning data
    df_clean = df.drop(columns=['Informant'])
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

    #convert to numpy
    data = df_clean.to_numpy(dtype=np.float32)
    return(data)

X = getData()
N, M = X.shape

with pm.Model() as model:
    #prior for competence, D_i [0.5,1]
    D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)

    #prior for consensus
    Z = pm.Bernoulli("Z", p=0.5, shape=M)

    #Broadcast D and Z
    D_re = D[:, None]
    p = Z * D_re + (1 - Z) * (1 - D_re)

    #likelihood
    X_obs = pm.Bernoulli("X_obs", p=p, observed=X)

    #sample from posterior
    trace = pm.sample(draws = 2000, tune=1000, chains = 4, cores = 4, target_accept=0.9, return_inferencedata= True )

#create and display summary
az_summary = az.summary(trace, hdi_prob=0.95)
print(az_summary)

#most and least competent informant
competence_summary = az_summary[az_summary.index.str.startswith("D[")]
competence_means = competence_summary['mean']
most_competent = competence_means.idxmax()
least_competent = competence_means.idxmin()
print(f"Most competent informant: {most_competent} (mean competence: {competence_means[most_competent]:.3f})")
print(f"Least competent informant: {least_competent} (mean competence: {competence_means[least_competent]:.3f})")

# plotting posterior for d
d_summary = az.summary(trace, var_names=["D"], hdi_prob=0.94)
means_d = d_summary['mean'].values
hdi_lower_d = d_summary['hdi_3%'].values
hdi_upper_d = d_summary['hdi_97%'].values
informants = [f"Inf {i+1}" for i in range(len(means_d))]
yerr_lower_d = np.clip(means_d - hdi_lower_d, 0, None)
yerr_upper_d = np.clip(hdi_upper_d - means_d, 0, None)

plt.figure(figsize=(10, 5))
plt.errorbar(informants, means_d, yerr=[yerr_lower_d, yerr_upper_d],
             fmt='o', capsize=5, label='Posterior Mean ± 94% HDI')

plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance Level (0.5)')
plt.xticks(rotation=45)
plt.ylim(0.45, 1.05)
plt.ylabel("Competence (D)")
plt.title("Posterior Estimates of Informant Competence")
plt.legend()
plt.tight_layout()
plt.show()

# plotting posterior for z
z_summary = az.summary(trace, var_names=["Z"], hdi_prob=0.94)
means = z_summary['mean'].values
hdi_lower = z_summary['hdi_3%'].values
hdi_upper = z_summary['hdi_97%'].values
questions = [f"Q{j+1}" for j in range(len(means))]
yerr_lower = np.clip(means - hdi_lower, 0, None)
yerr_upper = np.clip(hdi_upper - means, 0, None)


plt.figure(figsize=(10, 5))
plt.errorbar(questions, means, yerr=[yerr_lower, yerr_upper],
             fmt='o', capsize=5, label='Posterior Mean ± 94% HDI')

plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Threshold (0.5)')
plt.xticks(rotation=45)
plt.ylim(-0.05, 1.05)
plt.ylabel("P(Zj = 1)")
plt.title("Posterior Estimates of Consensus Answers Zj")
plt.legend()
plt.tight_layout()
plt.show()

#Comparison between naive and CCT answers
naive_means = X.mean(axis=0)
naive_answers = (naive_means >= 0.5).astype(int)

z_summary = az.summary(trace, var_names=["Z"])
cct_means = z_summary["mean"]
cct_answers = (cct_means >= 0.5).astype(int)

question_labels = [f"Q{j+1}" for j in range(20)]
comparison = pd.DataFrame({
    "Question": question_labels,
    "Naive": naive_answers,
    "CCT": cct_answers,
    "Same?": naive_answers == cct_answers
})

print(comparison)