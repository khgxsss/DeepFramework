from cores import Layer

import cores.functions as F
import cores.layers as L
import cores.utils

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return cores.utils.plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model): # Multi-Layer Perceptron
    def __init__(self, fc_output_sizes, activation=F.sigmoid): # full connect 완전연결 출력 크기, activation 함수
        super().__init__()
        self.activation = activation
        self.layers = []
        
        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)