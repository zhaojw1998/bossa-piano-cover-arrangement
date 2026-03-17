import os
import time
import torch
from model_blip2 import Blip2Qformer
from dataset import PIAST_Dataset, collate_fn_lmd, Ensemble_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import numpy as np


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank, world_size):
    ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    model = Blip2Qformer(load_pretrained=False)
    #num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'Number of trainable parameters: {num_params}')
    state_dict = torch.load("/data2/zhaojw/blip2_style_learning_WO_LMD_224notes_12bins_009_epoch.pt", map_location='cpu')
    for key in list(state_dict.keys()):
                new_key = key.replace('ln_audio', 'layer_norm')\
                            .replace('vision_proj', 'audio_proj')\
                            .replace('text_proj', 'symbo_proj')\
                            .replace('itm_head', 'binary_head')
                state_dict[new_key] = state_dict.pop(key)  
    model.load_state_dict(state_dict)
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)  
    model.eval();

    test_set = PIAST_Dataset(codec_dir="/data2/zhaojw/audio_midi_data/PIAST_codec.h5",
                            midi_dir="/data2/zhaojw/audio_midi_data/PIAST_octMIDI_12bins/",
                            split='test', debug_mode=False)
    """test_set = Ensemble_Dataset(debug_mode=False, split='test', load_num=15, pitch_order='high_to_low')"""
    test_loader = DataLoader(test_set, batch_size=32, drop_last=True, shuffle=True, collate_fn=lambda b: collate_fn_lmd(b, rank, augment=True)) 

    record = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            indices, loss = model.module.test_contrastive(*batch)
            rank = dist.get_rank()
            #print(indices)
            record.append(indices[:, 1].cpu().detach().numpy())
            break
    
    if rank == 0:   
        print(record)
        print(loss)
        mean_rank = np.mean(np.concatenate(record, axis=-1))
        print('mean reank:', mean_rank+1)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    world_size = torch.cuda.device_count()
    #print(world_size)
    mp.spawn(main, args=(world_size,), nprocs=world_size)