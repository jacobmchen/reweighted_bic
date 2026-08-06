import pickle
import matplotlib.pyplot as plt

num_experiments = 200

# get the sample sizes we tried
sample_sizes = [500, 1000, 2500, 5000, 7500, 10000]

# backdoor graph use (0.5, 0.14)
# frontdoor graph use 'best'

legend_loc=(0.5, 0.14)

filename='frontdoor_half_oracle_results.pkl'

pngname='frontdoor_half_oracle_graph.pdf'

# open the backdoor results
with open(filename, 'rb') as file:
    backdoor_results = pickle.load(file)

lin_reg = [ sample_results[0]/num_experiments for sample_results in backdoor_results ]
scad = [ sample_results[1]/num_experiments for sample_results in backdoor_results ]
alasso = [ sample_results[2]/num_experiments for sample_results in backdoor_results ]
bic_logn = [ sample_results[3]/num_experiments for sample_results in backdoor_results ]
bic_rtn = [ sample_results[4]/num_experiments for sample_results in backdoor_results ]
bic_n34 = [ sample_results[5]/num_experiments for sample_results in backdoor_results ]

# get the positions for the x-axis
xpos = range(len(sample_sizes))

plt.figure(figsize=(7,4.3))

# plot naive linear regression results
plt.plot(xpos, lin_reg, marker='o', linestyle='-', label='Naive Linear Regression')
plt.plot(xpos, scad, marker='o', linestyle='-', label='SCAD Regression')
plt.plot(xpos, alasso, marker='o', linestyle='-', label='Adaptive LASSO')
plt.plot(xpos, bic_logn, marker='o', linestyle='-', label=r'Reweighted BIC; $\log n$ Penalty')
plt.plot(xpos, bic_rtn, marker='o', linestyle='-', label=r'Reweighted BIC; $n^{1/2}$ Penalty')
plt.plot(xpos, bic_n34, marker='o', linestyle='-', label=r'Reweighted BIC; $n^{3/4}$ Penalty')

plt.xticks(xpos, sample_sizes)
plt.xlabel('Sample Size')
plt.ylabel(r'Percent Correct Choice of ${\bf A}^*$')
plt.title(r'Simulation Results for the Frontdoor Graph when Variance of $M$ is Known')

plt.legend(loc=legend_loc)

plt.savefig(pngname, format='pdf', bbox_inches='tight')

plt.show()
