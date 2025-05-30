
nnjl is neural network library written in Julia.

## Features
- Linear (Dense) Layer
- Convolutional Layer (not finished yet)
- Safe Dense Layer to/from JSON

## Installation
Clone the repository:
```
git clone https://github.com/OfflineBot/nnjl.git
```
Include it to your project:
```
cd <your-project>
julia
] activate .
] dev <path-to-nnjl>
] instantiate
```

## Usage
```
# Inlcude:
using nnjl

input = randn(2, 2)

# Create a dense layer with ReLU activation function
layer = DenseLayer(2, 4, relu)

# Forward the dense layer with the input data
forward!(layer, input)

# Safe the dense layer in json in "data/layer1.json"
write_json_dense_layer(layer, "layer1.json")
```


To see all functions/methods:
```
nnjl_info() # prints all the info about the library
```

