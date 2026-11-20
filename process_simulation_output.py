import sys
import pickle

num_experiments = 200

if len(sys.argv) > 1:
    file_header = sys.argv[1]
else:
    raise RuntimeError('User did not specify a file header')

pickle_file_name = 'pickle_files/' + file_header + '_results.pkl'

# create a list of lists to store results of the simulations
# the outer layer is a list corresponding to sample size
# the inner layer is whether each of the 6 methods selected the
# right model
# there are 6 sample sizes
# the 6 methods are: linear regression, SCAD regression,
# adaptive LASSO, reweighted BIC with logn penalty,
# reweighted BIC with n^(1/2) penalty, reweighted BIC with
# n^(3/4) penalty
results = []
for i in range(6):
    results.append([0]*12)

for i in range(num_experiments):
    output = []

    filename = file_header + '-' + str(i) + '.log'

    with open(filename, 'r') as file:
        for line in file:
            value = line.strip()
            output.append(value == 'True')

    for j in range(len(output)):
        # if output is true, the method at the sample size selected the right
        # model
        # the sample size corresponds j/12, and the method corresponds to i%12
        if output[j] == True:
            results[int(j/12)][j%12] += 1

# print results
sample_sizes = [500, 1000, 2500, 5000, 7500, 10000]
methods = ['naive regression', 'SCAD regression', 'ALASSO', 'BIC logn', 'BIC n^(1/2)', 'BIC n^(3/4)',
           'logn choose A2', 'logn no A1/A3', 'n^(1/2) choose A2', 'n^(1/2) no A1/A3', 'n^(3/4) choose A2', 'n^(3/4) no A1/A3']

for i in range(len(sample_sizes)):
    print('sample size', sample_sizes[i])
    for j in range(len(methods)):
        print(methods[j], 'percent correct/true:', results[i][j]/num_experiments)
    print()

print(results)

with open(pickle_file_name, 'wb') as file:
    pickle.dump(results, file)
