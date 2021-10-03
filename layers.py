import numpy as np 

class Dense:
    def __init__(self, units, input_shape = None, activation = None):
        self.units = units
        
        self.input_shape = input_shape 
        self.output_shape = (units, 1)

        self.activation = activation 

    def init_params(self):
        self.W = np.random.rand(self.input_shape[0], self.units)      #Layer Weight Initializing
        self.B = np.random.rand(self.units, 1)          #Layer Bias Initializing

    def getShapes(self): 
        print("W:", self.W.shape)
        print("B:", self.B.shape)

    def forwardPass(self, input_tensor):
        assert input_tensor.shape[1] == 1
        assert len(input_tensor.shape) == 2 

        out = self.W.T.dot(input_tensor) + self.B 

        return out if self.activation == None else self.activation(out)
            