is_simple_core = True

if is_simple_core:
    from core.core_simple import Variable
    from core.core_simple import Function
    from core.core_simple import using_config
    from core.core_simple import no_grad
    from core.core_simple import as_array
    from core.core_simple import as_variable
else:
    from core.core import Variable
    from core.core import Function
    from core.core import using_config
    from core.core import no_grad
    from core.core import as_array
    from core.core import as_variable

