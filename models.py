import numpy as np 
import layers 

class Model:
    def __init__(self):
        self.layer_graph = [] 

    def add(self, lyr): 
        # assert type(lyr) == layers.object

        if lyr.input_shape == None:
            lyr.input_shape = self.layer_graph[-1].output_shape

        lyr.init_params()  
        self.layer_graph.append(lyr) 

    def forwardPass(self, input_tensor):
        x = input_tensor 
        for lyr in self.layer_graph:
            x = lyr.forwardPass(x) 
        return x 