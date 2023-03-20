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
    from cores.core import Variable
    from cores.core import Function
    from cores.core import using_config
    from cores.core import no_grad

    from cores.functions import sin, cos, tanh, reshape

from cores.utils import plot_dot_graph