from kernel.Linear_kernel import Linear_kernel
from kernel.SE_kernel_module import SE_kernel
from kernel.RQ_kernel import RQ_kernel
from kernel.Combine_kernel import Combine_kernel
from kernel.Periodic_kernel import Periodic_kernel

def kernel_generator(kernel_name, config_dict):
    kernel_list = {
        'linear': Linear_kernel,
        'SE': SE_kernel,
        'RQ': RQ_kernel,
        'Periodic': Periodic_kernel,
        'Local_Periodic': Periodic_kernel,
    }
    
    if kernel_name in kernel_list:
        return kernel_list[kernel_name](**config_dict)

    elif kernel_name == 'Combine_Linear_SE':
        cb_list = [Linear_kernel(**config_dict['Linear']), SE_kernel(**config_dict['SE'])]
        return Combine_kernel(cb_list)

    else:
        assert False, "kernel not found for {}".format(kernel_name)