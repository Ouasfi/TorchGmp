# Secure MLP Based on HE in PyTorch

A pytorch  implementation of Secure Multilayer Perceptron Based on Homomorphic Encryption (https://link.springer.com/chapter/10.1007/978-3-030-11389-6_24#chapter-info)


# Usage
- Inspect the C++  extensions in the `cpp/`  folder.

## Integer MPL
- You can train over integers by going into the `cpp/` folder and running `python model_int.py`. This model takes around 2à epochs to converge. 
## Encrypted MLP
- Build C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and executing `python setup.py install`,

Or, 

- JIT-compile C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and calling `python jit.py`, which will JIT-compile the extension and load it,
- Run `python model_enc.py`. This model is unstable and may need some tuning of the hyper-parameters to converge.

# 0rganization
This repo is organized in the following way:
- `cuda/` folder is a placeholder for the cuda accelerated version of the cpp module.
- `cpp/`  is the main focus of this implementation. It contains the following files:
    - `encrypted_ops.py`: Implements the class of `EncryptedTensor`. It handles the operations over HE encrypted tensors using the `phe` module .
    - `integer_ops.py`: Implements the class of `IntegerTensor`. It handles the operations over torch.int64 tensors.
    - `phe.py`: Implements the operations over HE encrypted tensors. The classes `P1` and `P2` correspond to the non-colluding semi-honest servers described in the paper. 
    - `model_int.py`: pytorch model for training over integers. 
    - `model_enc.py`: pytorch model for encrypted training. 
    - `model_th.py` : A baseline model. In constrast to the previous ones, here the gradients are accumulated by iterating over the samples of the batch resulting in a slower training. 
    - `lltm.cpp`:  C++ implementation of the operations over tensors. It's based on `gmp` for arbitrary precision arithmetic  and on `libhcs` for paillier HE. 


## Authors

[Amine Ouasfi](https://github.com/ouasfi)

[Iheb Kotorsi]
