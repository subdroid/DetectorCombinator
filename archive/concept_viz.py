import matplotlib.pyplot as plt
import numpy as np

import torch

# Set the number of activations and distributions
num_activations = 100
num_distributions = 10

# Set up a figure
fig, ax = plt.subplots(figsize=(8, 6))
# Define a colormap based on the number of distributions
# cmap = plt.get_cmap('viridis', num_distributions)

Activations = [] 

for i in range(num_distributions):
    # Randomly generate mean and standard deviation for each distribution
    mean = np.random.uniform(10, 10000)
    std_dev = np.random.uniform(500, 2000)

    # Sample activations from a normal distribution
    activations = np.random.normal(mean, std_dev, size=num_activations)
    
    Activations.append(activations)

    # Create x values to represent the indices of activations
    indices = np.arange(len(activations))

    # Scatter plot for each distribution with colormap*num_activat
    # sc = ax.scatter(indices, activations, marker='o', s=2, color=cmap(i*3), label=f'Neuron {i+1}')
    sc = ax.scatter(indices, activations, marker='o', s=5, label=f'Neuron {i+1}')
   
    # Connect the points for each distribution with a line
    # ax.plot(indices, activations, marker='o', markersize=3, linestyle='-', color=cmap(i), label=f'Neuron {i+1}')


# Set labels and title
ax.set_xlabel('Sentence Index')
ax.set_ylabel('Activation Value')
ax.set_title(f'Scatter Plots of Activations for {num_distributions} different neurons')

# Place legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the plot
plt.savefig("activation.png", bbox_inches='tight')


plt.close()
plt.clf()

fig, ax = plt.subplots(figsize=(8, 6))

for i,activations in enumerate(Activations):
    indices = np.arange(len(activations))
    ax.plot(indices, activations, marker='o', label=f'Neuron {i+1}')
    ax.fill_between(indices, 0, activations, alpha=0.2)

# Set labels and title
ax.set_xlabel('Sentence Index')
ax.set_ylabel('Activation Value')
ax.set_title(f'Distribution of Activations for {num_distributions} different neurons')

# Place legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the plot
plt.savefig("activation_dist.png", bbox_inches='tight')


plt.close()
plt.clf()



activations = np.array(Activations)
activations_tensor = torch.tensor(activations, dtype=torch.float64)

# Apply softmax across the first dimension
activations_softmax = torch.nn.functional.softmax(activations_tensor, dim=0).numpy()

expectations = activations * activations_softmax

fig, ax = plt.subplots(figsize=(8, 6))

for i,activations in enumerate(expectations):
    indices = np.arange(len(activations))
    sc = ax.scatter(indices, activations, marker='o', s=5, label=f'Neuron {i+1}')
    
# Set labels and title
ax.set_xlabel('Sentence Index')
ax.set_ylabel('Activation Value')
ax.set_title(f'Scatter Plots of Expectations for {num_distributions} different neurons')

# Place legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the plot
plt.savefig("expectation.png", bbox_inches='tight')

plt.close()
plt.clf()

fig, ax = plt.subplots(figsize=(8, 6))

for i,activations in enumerate(expectations):
    indices = np.arange(len(activations))
    ax.plot(indices, activations, marker='o', label=f'Neuron {i+1}')
    ax.fill_between(indices, 0, activations, alpha=0.2)
    
# Set labels and title
ax.set_xlabel('Sentence Index')
ax.set_ylabel('Activation Value')
ax.set_title(f'Distribution of Expectations for {num_distributions} different neurons')

# Place legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the plot
plt.savefig("expectation_dist.png", bbox_inches='tight')
