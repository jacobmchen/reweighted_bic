import sys
import pickle
import matplotlib.pyplot as plt

if len(sys.argv) > 2:
    frontdoor_title = sys.argv[1]
    backdoor_title = sys.argv[2]
else:
    raise RuntimeError('User did not specify the files to plot for the weights.')

# read the data on the frontdoor weights
with open('pickle_files/' + frontdoor_title + '.pkl', 'rb') as file:
    weights = pickle.load(file)

# convert the weights to a numpy array
weights = weights.to_numpy()

plt.figure(figsize=(7,4.3))

plt.hist(weights, bins=30, color='skyblue', edgecolor='black')

plt.title('Histogram of Weights in the Frontdoor Model')
plt.xlabel('Values')
plt.ylabel('Frequency')

plt.savefig('figures/' + frontdoor_title + '.pdf', format='pdf', bbox_inches='tight')

# read the data on the backdoor weights
with open('pickle_files/' + backdoor_title + '.pkl', 'rb') as file:
    weights = pickle.load(file)

# convert the weights to a numpy array
weights = weights.to_numpy()

plt.figure(figsize=(7,4.3))

plt.hist(weights, bins=30, color='skyblue', edgecolor='black')

plt.title('Histogram of Weights in the Backdoor Model')
plt.xlabel('Values')
plt.ylabel('Frequency')

plt.savefig('figures/' + backdoor_title + '.pdf', format='pdf', bbox_inches='tight')

plt.show()

plt.close()
