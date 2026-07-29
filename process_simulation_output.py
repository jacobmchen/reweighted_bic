num_experiments = 1

file_header = 'backdoor'

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

print(results)
