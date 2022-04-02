import importlib
import os
import os.path as osp

from utils.registry import LOSS_REGISTRY, LR_SCHEDULER_REGISTRY

loss_sch_folder = osp.dirname(osp.abspath(__file__))
loss_sch_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in os.listdir(loss_sch_folder)
    if v.endswith(".py")
]
# import all the arch modules
_loss_sch_modules = [
    importlib.import_module(f"loss_scheduler.{file_name}") for file_name in loss_sch_filenames
]

def build_loss(loss_opt):
    loss_type = loss_opt.pop("type")
    loss = LOSS_REGISTRY.get(loss_type)(**loss_opt)
    return loss

def build_scheduler(optimizer, scheduler_opt):
    scheduler_type = scheduler_opt.pop("type")
    scheduler = LR_SCHEDULER_REGISTRY.get(scheduler_type)(optimizer, **scheduler_opt)
    return scheduler
