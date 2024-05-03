# Comparative Analysis of Neural Network-Based Methods for Wave Propagation Modeling
 
This repository contains a collection of notebooks that demonstrate various deep learning approaches to modeling the wave equation. The following notebooks are included:

1. [Simple Function Approximation with PyTorch](https://github.com/oscar-rincon/NeuralNetworks-WavePropagation/blob/main/notebooks/1_Simple_Function_Aproximation.ipynb)
2. [Approximation of the 1-D Wave Equation With Neural Networks Using PyTorch](https://github.com/oscar-rincon/NeuralNetworks-WavePropagation/blob/main/notebooks/2_1D_Wave_Equation_Approximation_with_NNs.ipynb)
3. [Approximation of the 1-D Wave Equation with Physics-Informed Neural Networks (PINNs) Using PyTorch](https://github.com/oscar-rincon/NeuralNetworks-WavePropagation/blob/main/notebooks/3_Approximation_with_PINNs.ipynb)
4. [NNs-Based 1D Acoustic Wave Simulation with a Source Term](https://github.com/oscar-rincon/NeuralNetworks-WavePropagation/blob/main/notebooks/4_1D_Acoustic_Wave_Source_NNs.ipynb)
5. [PINNs-Based 1D Acoustic Wave Simulation with a Source Term](https://github.com/oscar-rincon/NeuralNetworks-WavePropagation/blob/main/notebooks/5_1D_Acoustic_Wave_Source_PINNs.ipynb)


# Instalation

We suggest setting up a new Python environment, for example:

 ```
conda create --name comparative-nn-wave-env
conda activate comparative-nn-wave-env
 ```

then cloning this repository:

 ```
git clone https://github.com/oscar-rincon/NeuralNetworks-WavePropagation.git
 ```

and running this command in the base  `NeuralNetworks-WavePropagation/` directory:

 ```
 conda env update --name comparative-nn-wave-env --file comparative_nn_wave_env.yaml
 ```
