import sys
import pickle
import matplotlib.pyplot as plt

if len(sys.argv) > 2:
    file_header = sys.argv[1]
    title_words = sys.argv[2]
else:
    raise RuntimeError('User did not specify a file header and words for graph title.')

filename = 'pickle_files/' + file_header + '_results.pkl'
pngname = 'figures/' + file_header + '_graph.pdf' 
pngname_bic_astar = 'figures/' + file_header + '_graph_bic_astar.pdf'

num_experiments = 200

# get the sample sizes we tried
sample_sizes = [500, 1000, 2500, 5000, 7500, 10000]

# backdoor graph use (0.5, 0.14)
# frontdoor graph use 'best'

legend_loc=(0.03, 0.14)

# open the results
with open(filename, 'rb') as file:
    results = pickle.load(file)

lin_reg = [ sample_results[0]/num_experiments for sample_results in results ]
scad = [ sample_results[1]/num_experiments for sample_results in results ]
alasso = [ sample_results[2]/num_experiments for sample_results in results ]
bic_logn = [ sample_results[3]/num_experiments for sample_results in results ]
bic_rtn = [ sample_results[4]/num_experiments for sample_results in results ]
bic_n34 = [ sample_results[5]/num_experiments for sample_results in results ]

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
plt.title('Simulation Results for ' + title_words.replace('_', ' '))

plt.legend(loc=legend_loc)

plt.savefig(pngname, format='pdf', bbox_inches='tight')

logn_A2 = [ sample_results[6]/num_experiments for sample_results in results ]
logn_no_A1_A3 = [ sample_results[7]/num_experiments for sample_results in results ]
n12_A2 = [ sample_results[8]/num_experiments for sample_results in results ]
n12_no_A1_A3 = [ sample_results[9]/num_experiments for sample_results in results ]
n34_A2 = [ sample_results[10]/num_experiments for sample_results in results ]
n34_no_A1_A3 = [ sample_results[11]/num_experiments for sample_results in results ]

# get the positions for the x-axis
xpos = range(len(sample_sizes))

plt.figure(figsize=(7,4.3))

# plot naive linear regression results
plt.plot(xpos, logn_A2, marker='o', linestyle='-', label=r'$\log n$: Choose $A_2$')
plt.plot(xpos, logn_no_A1_A3, marker='o', linestyle='-', label=r'$\log n$: Did not choose $A_1$/$A_3$')
plt.plot(xpos, n12_A2, marker='o', linestyle='-', label=r'$n^{1/2}$: Choose $A_2$')
plt.plot(xpos, n12_no_A1_A3, marker='o', linestyle='-', label=r'$n^{1/2}$: Did not choose $A_1$/$A_3$')
plt.plot(xpos, n34_A2, marker='o', linestyle='-', label=r'$n^{3/4}$: Choose $A_2$')
plt.plot(xpos, n34_no_A1_A3, marker='o', linestyle='-', label=r'$n^{3/4}$: Did not choose $A_1$/$A_3$')

plt.xticks(xpos, sample_sizes)
plt.xlabel('Sample Size')
plt.ylabel(r'Percent of Event')
plt.title(r'Reasons for wrong choice of ${\bf A}^*$ in ' + title_words.replace('_', ' '))

plt.legend(loc=legend_loc)

plt.savefig(pngname_bic_astar, format='pdf', bbox_inches='tight')

plt.show()

plt.close()

