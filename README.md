# Comparative Analysis of Neural Network-Based Methods for Wave Propagation Modeling
 
This repository contains a set of notebooks that demonstrate different deep learning techniques for simulating wave propagation, as described by the wave equation. The included notebooks are as follows:


1. [Simple Function Approximation with PyTorch](https://github.com/oscar-rincon/NeuralNetworks-WavePropagation/blob/main/notebooks/1_Simple_Function_Aproximation.ipynb)
2. [Approximation of the 1D Wave Equation With Neural Networks Using PyTorch](https://github.com/oscar-rincon/NeuralNetworks-WavePropagation/blob/main/notebooks/2_1D_Wave_Equation_Approximation_with_NNs.ipynb)
3. [Approximation of the 1D Wave Equation with Physics-Informed Neural Networks (PINNs) Using PyTorch](https://github.com/oscar-rincon/NeuralNetworks-WavePropagation/blob/main/notebooks/3_Approximation_with_PINNs.ipynb)
4. [NNs-Based 1D Acoustic Wave Simulation with a Source Term](https://github.com/oscar-rincon/NeuralNetworks-WavePropagation/blob/main/notebooks/4_1D_Acoustic_Wave_Source_NNs.ipynb)
5. [PINNs-Based 1D Acoustic Wave Simulation with a Source Term](https://github.com/oscar-rincon/NeuralNetworks-WavePropagation/blob/main/notebooks/5_1D_Acoustic_Wave_Source_PINNs.ipynb)

# Installation

We recommend setting up a new Python environment. You can do this by running the following commands:

 ```
 conda create --name comparative-nn-wave-env
 conda activate comparative-nn-wave-env
 ```

Next, clone this repository by using the command:

 ```
git clone https://github.com/oscar-rincon/NeuralNetworks-WavePropagation.git
 ```

Finally, go to the `NeuralNetworks-WavePropagation/` folder and run the following command to install the necessary dependencies:

 ```
 conda env update --name comparative-nn-wave-env --file comparative_nn_wave_env.yaml
 ```

To verify the packages installed in your `comparative-nn-wave-env` conda environment, you can use the following command:

 ```
 conda list -n comparative-nn-wave-env
 ```

 