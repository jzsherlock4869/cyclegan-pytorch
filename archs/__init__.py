import importlib
import os
import os.path as osp

from utils.registry import ARCH_REGISTRY

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in os.listdir(arch_folder)
    if v.endswith(".py")
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f"archs.{file_name}") for file_name in arch_filenames
]


def build_network(net_opt):
    which_network = net_opt["which_network"]
    net = ARCH_REGISTRY.get(which_network)(**net_opt["setting"])
    print(f'Network [{net.__class__.__name__}] is created.')

    return net

