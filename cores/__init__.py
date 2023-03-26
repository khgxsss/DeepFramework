is_simple_core = False

if is_simple_core:
    from cores.core_simple import Variable
    from cores.core_simple import Function
    from cores.core_simple import using_config
    from cores.core_simple import no_grad
    from cores.core_simple import as_array
    from cores.core_simple import as_variable
    from cores.core_simple import sin
    from cores.core_simple import square
    from cores.core_simple import exp
    from cores.core_simple import numerical_diff
else:
    from cores.core import Function, Parameter, Variable
    from cores.core import as_array, as_variable, no_grad, using_config
    from cores.layers import Layer
    from cores.models import Model
    from cores.dataloaders import DataLoader, SeqDataLoader
    from cores.datasets import Dataset

    import cores.cuda
    import cores.dataloaders
    import cores.datasets
    import cores.functions
    import cores.layers
    import cores.models
    import cores.optimizers
    import cores.transforms
    import cores.utils
    
from cores.utils import plot_dot_graph