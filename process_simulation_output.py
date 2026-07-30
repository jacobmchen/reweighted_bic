import pickle

num_experiments = 200

file_header = 'frontdoor'

# create a list of lists to store results of the simulations
# the outer layer is a list corresponding to sample size
# the inner layer is whether each of the 6 methods selected the
# right model
# the 6 methods are: linear regression, SCAD regression,
# adaptive LASSO, reweighted BIC with logn penalty,
# reweighted BIC with n^(1/2) penalty, reweighted BIC with
# n^(3/4) penalty
results = []
for i in range(6):
    results.append([0]*6)

for i in range(num_experiments):
    output = []

    filename = file_header + '-' + str(i) + '.log'

    with open(filename, 'r') as file:
        for line in file:
            value = line.strip()
            output.append(value == 'True')

    for i in range(len(output)):
        # if output is true, the method at the sample size selected the right
        # model
        # the sample size corresponds i/6, and the method corresponds to i%6
        if output[i] == True:
            results[int(i/6)][i%6] += 1

# print results
sample_sizes = [500, 1000, 2500, 5000, 7500, 10000]
methods = ['naive regression', 'SCAD regression', 'ALASSO', 'BIC logn', 'BIC n^(1/2)', 'BIC n^(3/4)']

for i in range(len(sample_sizes)):
    print('sample size', sample_sizes[i])
    for j in range(len(methods)):
        print(methods[j], 'percent correct:', results[i][j]/num_experiments)
    print()

print(results)

with open('frontdoor_results_coef3.5.pkl', 'wb') as file:
    pickle.dump(results, file)
