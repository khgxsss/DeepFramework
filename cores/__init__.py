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

    from cores.operators import Setup_Variable
    from cores.operators import add
    from cores.operators import mul
    from cores.operators import neg
    from cores.operators import sub
    from cores.operators import div
    from cores.operators import pow

    from cores.functions import sin
    from cores.functions import square
    from cores.functions import exp

    from cores.d_structures import as_array
    from cores.d_structures import as_variable

from cores.utils import plot_dot_graph