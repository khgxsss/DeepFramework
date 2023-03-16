is_simple_core = True

if is_simple_core:
    from core.core_simple import Variable
    from core.core_simple import Function
    from core.core_simple import using_config
    from core.core_simple import no_grad
    from core.core_simple import as_array
    from core.core_simple import as_variable
    from core.core_simple import sin
    from core.core_simple import square
    from core.core_simple import exp
else:
    from core.core import Variable
    from core.core import Function
    from core.core import using_config
    from core.core import no_grad
    from core.core import as_array
    from core.core import as_variable

from core.utils import plot_dot_graph