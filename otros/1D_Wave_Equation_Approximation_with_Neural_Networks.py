# Import NumPy for numerical operations
import numpy as np

# Import PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim

# Import Matplotlib for plotting
import matplotlib.pyplot as plt

# Import a utility module for additional plotting functions
import utils_plots

# Import the time module to time our training process
import time

# Define a function for the analytical solution of the 1D wave equation
def analytic_sol_func(t, x):
    C = 1
    return sum([(8 / (k**3 * np.pi**3)) * np.sin(k * np.pi * x) * np.cos(C * k * np.pi * t) for k in range(1, 100, 2)])

# Generate training data in NumPy
x_np = np.linspace(0, 1, 100)  # x data (numpy array), shape=(100,)
t_np = np.linspace(0, 1, 100)  # t data (numpy array), shape=(100,)

# Create a grid of x and t values
x_grid, t_grid = np.meshgrid(x_np, t_np) # x and t data (numpy array), shape=(100, 100)

# Calculate u values using the analytical solution function
u_grid = analytic_sol_func(t_grid,x_grid) # u data (numpy array), shape=(100, 100)

# Create a figure for the plot
fig = plt.figure(figsize=(3.2, 2.2))

# Plot the u values as a function of t and x
plt.contourf(t_grid,x_grid,u_grid, origin='lower', extent=(0, 1, 0, 1), levels=20,cmap=utils_plots.cmap_)
plt.colorbar(label='$u$')
plt.contour(t_grid, x_grid, u_grid, levels=20, colors='white', alpha=0.2, linestyles='solid')
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.title('Analytical solution of the 1D wave equation')

# Save the plot as a PDF file in the 'imgs' directory
fig.savefig('imgs/1D_Wave_Equation_Analytical_Solution.pdf', format='pdf')

# Display the plot
plt.show()

# Conversion of the grid data to PyTorch tensors
x = torch.from_numpy(x_grid).float().unsqueeze(-1)
t = torch.from_numpy(t_grid).float().unsqueeze(-1)
u = torch.from_numpy(u_grid).float().unsqueeze(-1)

# Concatenation of x and y to form the input data
input_data = torch.cat((x, t), dim=-1)

# Define a neural network class with three fully connected layers
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 50)
        self.layer2 = nn.Linear(50, 50)
        self.output_layer = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.output_layer(x)
        return x
    
# Create an instance of the neural network
neural_net = NeuralNetwork()

# Define an optimizer (Adam) for training the network
optimizer = optim.Adam(neural_net.parameters(), lr=0.01)

# Define a loss function (Mean Squared Error) for training the network
loss_func = nn.MSELoss()

# Initialize a list to store the loss values
loss_values = []

# Start the timer
start_time = time.time()

# Training the neural network
for i in range(1000):
    prediction = neural_net(input_data)     # input x and predict based on x
    loss = loss_func(prediction, u)     # must be (1. nn output, 2. target)
    
    # Append the current loss value to the list
    loss_values.append(loss.item())
    
    if i % 100 == 0:  # print every 100 iterations
        print(f"Iteration {i}: Loss {loss.item()}")
    
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()

# Stop the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")

# Save a summary of the training process to a text file
with open("training_summary.txt", "w") as file:
    file.write(f"Training time: {elapsed_time} seconds\n")
    file.write(f"Number of iterations: {len(loss_values)}\n")
    file.write(f"Initial loss: {loss_values[0]}\n")
    file.write(f"Final loss: {loss_values[-1]}\n")
    file.write(f"Neural network architecture: {neural_net}\n")
    file.write(f"Optimizer used: {type(optimizer).__name__}\n")
    file.write(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")

# Create a figure for the plot
fig = plt.figure(figsize=(3.5, 2.5))

# Plot the loss values recorded during training
plt.plot(loss_values, color='gray', linewidth=2)

# Set the labels for the x and y axes
plt.xlabel('Iteration')
plt.ylabel('Loss')

# Set the title for the plot
plt.title('Training Progress')

# Display the grid
plt.grid(True)

# Save the plot as a PDF file in the 'imgs' directory
fig.savefig('imgs/Training_Progress_Loss_Per_Iteration.pdf', format='pdf')

# Display the plot
plt.show()

# In the following code block, we save the trained model's parameters to a file. We then initialize a new instance of the neural network and load the saved parameters into this new instance. This allows us to reuse the trained model without having to retrain it.

# Save the trained model's parameters to a file
torch.save(neural_net.state_dict(), 'models/1D_Wave_Equation_NN_Model.pth')

# Initialize a new instance of the neural network
new_neural_net = NeuralNetwork()

# Load the saved parameters into the new instance of the neural network
new_neural_net.load_state_dict(torch.load('models/1D_Wave_Equation_NN_Model.pth'))

# Ensure the new neural network is in evaluation mode
new_neural_net.eval()

# Generate predictions using the neural network
u_pred = neural_net(input_data).detach().numpy().reshape(x_grid.shape)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(6, 2))

# Plot the predicted u values
im1 = axs[0].contourf(t_grid,x_grid,u_pred, origin='lower', levels=20,cmap=utils_plots.cmap_)
axs[0].set_title('Predicted')
axs[0].set_xlabel('$x$')
axs[0].set_ylabel('$t$')
fig.colorbar(im1, ax=axs[0], label='$u$')
axs[0].contour(t_grid, x_grid, u_pred, levels=20, colors='white', alpha=0.2, linestyles='solid')

# Plot the difference between the predicted and analytical u values
im2 = axs[1].contourf(t_grid,x_grid,u_pred-u_grid, origin='lower', levels=20,cmap=utils_plots.cmap_)
axs[1].contour(t_grid, x_grid, u_pred-u_grid, levels=20, colors='white', alpha=0.2, linestyles='solid')
axs[1].set_title('Difference')
axs[1].set_xlabel('$x$')
axs[1].set_ylabel('$t$')
fig.colorbar(im2, ax=axs[1], label='$u$')

# Save the plot as a PDF file in the 'imgs' directory
fig.savefig('imgs/Predicted_vs_Difference_1D_Wave_Equation.pdf', format='pdf')

# Display the plot
plt.show()