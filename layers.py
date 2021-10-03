import numpy as np 


class Dense:
    def __init__(self, units, input_shape = None):
        self.units = units
        
        assert input_shape[1] == 1
        assert len(input_shape) == 2 

        self.input_shape = input_shape 
        self.output_shape = (units, 1)

        self.W = np.random.rand(input_shape[0], units)      #Layer Weight Initializing
        self.B = np.random.rand(units, 1)          #Layer Bias Initializing

    def getShapes(self): 
        print("W:", self.W.shape)
        print("B:", self.B.shape)

    def forwardPass(self, input_tensor):
        assert input_tensor.shape[1] == 1
        assert len(input_tensor.shape) == 2 

        return self.W.T.dot(input_tensor) + self.B
``