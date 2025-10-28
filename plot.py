import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
time_steps = [1, 2, 4, 8]
accuracy_gs_nomad = [0.708, 0.732, 0.869, 0.949]
accuracy_gs_ours = [0.407, 0.922, 0.923, 0.921]
accuracy_recon_nomad = [0.75, 0.78, 0.81, 0.85]
accuracy_recon_ours = [0.418, 0.947, 0.944, 0.935]

# Creating a DataFrame for seaborn
data = pd.DataFrame({
    'Time Steps': time_steps * 4,
    'Accuracy': accuracy_gs_nomad + accuracy_gs_ours + accuracy_recon_nomad + accuracy_recon_ours,
    'Dataset': ['Go Stanford'] * 8 + ['RECON'] * 8,
    'Algorithm': ['NoMAD'] * 4 + ['Ours'] * 4 + ['NoMAD'] * 4 + ['Ours'] * 4
})

# Creating the plot
sns.lineplot(data=data, x='Time Steps', y='Accuracy', hue='Algorithm', style='Dataset', markers=True, 
             dashes={'Go Stanford': '', 'RECON': (2, 2)}, linewidth=2.5)

# Set custom linewidth for Dataset 1 to make it bold
plt.gca().lines[0].set_linewidth(4)  # Bold for Dataset 1 (Algo 1)
plt.gca().lines[1].set_linewidth(4)  # Bold for Dataset 1 (Algo 2)

# Adding labels and title
plt.xlabel('Time Steps')
plt.ylabel('Action Accuracy')
plt.title('Action Accuracy for Different Algorithms and Datasets')

# Show plot
plt.show()