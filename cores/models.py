from cores.layers import Layer, Linear
from cores.functions import sigmoid
from cores.utils import plot_dot_graph

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model): # Multi-Layer Perceptron
    def __init__(self, fc_output_sizes, activation=sigmoid): # full connect 완전연결 출력 크기, activation 함수
        super().__init__()
        self.activation = activation
        self.layers = []
        
        for i, out_size in enumerate(fc_output_sizes):
            layer = Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)