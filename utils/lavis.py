import torch
import torch.distributed as dist

from dataclasses import dataclass
from typing import Optional

from transformers.modeling_outputs import ModelOutput



def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



@dataclass
class BlipOutputOctMIDI(ModelOutput):

    loss: Optional[torch.FloatTensor] = None

    loss_contrastive: Optional[torch.FloatTensor] = None

    loss_matching: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None

    loss_lm_bar: Optional[torch.FloatTensor] = None

    loss_lm_pos: Optional[torch.FloatTensor] = None

    loss_lm_ins: Optional[torch.FloatTensor] = None

    loss_lm_pch: Optional[torch.FloatTensor] = None

    loss_lm_dur: Optional[torch.FloatTensor] = None

    loss_lm_vel: Optional[torch.FloatTensor] = None

    loss_lm_ts: Optional[torch.FloatTensor] = None

    loss_lm_tmp: Optional[torch.FloatTensor] = None

    loss_ml: Optional[torch.FloatTensor] = None

    loss_ml_bar: Optional[torch.FloatTensor] = None

    loss_ml_pos: Optional[torch.FloatTensor] = None

    loss_ml_ins: Optional[torch.FloatTensor] = None

    loss_ml_pch: Optional[torch.FloatTensor] = None

    loss_ml_dur: Optional[torch.FloatTensor] = None

    loss_ml_vel: Optional[torch.FloatTensor] = None

    loss_ml_ts: Optional[torch.FloatTensor] = None

    loss_ml_tmp: Optional[torch.FloatTensor] = None
