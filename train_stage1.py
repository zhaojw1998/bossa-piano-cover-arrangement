import os
import time
import torch
from torch import optim
from model_blip2 import Blip2Qformer
from dataset import Ensemble_Dataset, collate_fn_musicbert
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from utils.scheduler import OptimizerSchedulerWithWarmUp
from utils.training import SummaryWriters, LogPathManager, epoch_time
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank, world_size, log_path_mng, VERBOSE, MODEL_NAME):
    #print('rank:', rank)
    ddp_setup(rank, world_size)

    BATCH_SIZE = 32
    N_EPOCH = 10
    CLIP = 1
    LR = 1e-4
    WARMUP_STEP = 1000

    if VERBOSE:
        N_EPOCH=5
        LR = 1e-3
        WARMUP_STEP=10

    torch.cuda.set_device(rank)
    
    model = Blip2Qformer()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params}')
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False) 

    torch.cuda.empty_cache()

    """train_set = LMD_Match_Dataset(codec_dir="/data1/zhaojw/lmd_aligned/lmd_audio.h5",
                                  midi_dir="/data1/zhaojw/lmd_aligned/lmd_processed_12bins/", 
                                  match_score_dir="/data1/zhaojw/lmd_aligned/match_scores.json", 
                                  split='train', debug_mode=VERBOSE, load_num=100)"""
    train_set = Ensemble_Dataset(debug_mode=VERBOSE, split='train', load_num=30, pitch_order='high_to_low')
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, drop_last=True, shuffle=False, collate_fn=lambda b: collate_fn_musicbert(b, rank, augment=True, deperf=True), sampler=DistributedSampler(train_set)) 

    """val_set = LMD_Match_Dataset(codec_dir="/data1/zhaojw/lmd_aligned/lmd_audio.h5",
                                  midi_dir="/data1/zhaojw/lmd_aligned/lmd_processed_12bins/", 
                                  match_score_dir="/data1/zhaojw/lmd_aligned/match_scores.json", 
                                  split='validation', debug_mode=VERBOSE, load_num=100)"""
    val_set = Ensemble_Dataset(debug_mode=VERBOSE, split='validation', load_num=15, pitch_order='high_to_low')
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, drop_last=True, shuffle=False, collate_fn=lambda b: collate_fn_musicbert(b, rank, augment=True, deperf=True), sampler=DistributedSampler(val_set))
    print(f'Dataset loaded. {len(train_loader)} samples for train and {len(val_loader)} samples for validation.')

    scaler = torch.cuda.amp.GradScaler()
    #optimizer = optim.Adam(model.parameters(), lr=LR)
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=[0.9, 0.999], weight_decay=0.05)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-2, end_factor=1, total_iters=WARMUP_STEP)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader)*N_EPOCH-WARMUP_STEP, eta_min=1e-5)
    #scheduler = MinExponentialLR(optimizer, gamma=0.99998, minimum=1e-5)
    
    #optimizer_scheduler = OptimizerScheduler(optimizer, scheduler, CLIP)
    optimizer_scheduler = OptimizerSchedulerWithWarmUp(optimizer, warmup_scheduler, scheduler, CLIP, WARMUP_STEP)
    

    writer_names = ['loss', 'loss_itc', 'loss_itm', 'loss_lm',
                    'loss_lm_bar', 'loss_lm_pos', 'loss_lm_ins', 'loss_lm_pch', 'loss_lm_dur', 'loss_lm_vel', 'loss_lm_ts', 'loss_lm_tmp']
    scheduler_writer_names = ['lr']
    
    if rank == 0:
        tags = {'loss': None}
        loss_writers = SummaryWriters(writer_names, tags, log_path_mng.writer_path)
        tags = {'scheduler': None}
        scheduler_writers = SummaryWriters(scheduler_writer_names, tags, log_path_mng.writer_path)
    else:
        loss_writers = None
        scheduler_writers = None
        VERBOSE = False


    for n_epoch in range(N_EPOCH):
        start_time = time.time()
        train_loader.sampler.set_epoch(n_epoch)
        print(f'Training epoch {n_epoch}')
        train_loss = train(model, train_loader, optimizer_scheduler, writer_names, loss_writers, scheduler_writers, n_epoch=n_epoch, VERBOSE=VERBOSE, scaler=scaler)['loss']
        print(f'Validating epoch {n_epoch}')
        val_loss = val(model, val_loader, writer_names, loss_writers, n_epoch=n_epoch, VERBOSE=VERBOSE)['loss']
        end_time = time.time()

        if rank == 0:
            torch.save(model.module.state_dict(), log_path_mng.epoch_model_path(f'{MODEL_NAME}_{str(n_epoch).zfill(3)}'))
        
        epoch_report(start_time, end_time, train_loss, val_loss, n_epoch)

    destroy_process_group()



def accumulate_loss_dic(writer_names, loss_dic, loss_items):
        assert len(writer_names) == len(loss_items)
        for key, val in zip(writer_names, loss_items):
            loss_dic[key] += loss_items[val].item()
        return loss_dic

def write_loss_to_dic(writer_names, loss_items):
    loss_dic = {}
    assert len(writer_names) == len(loss_items)
    for key, val in zip(writer_names, loss_items):
        loss_dic[key] = loss_items[val].item()
    return loss_dic

def init_loss_dic(writer_names):
        loss_dic = {}
        for key in writer_names:
            loss_dic[key] = 0.
        return loss_dic

def average_epoch_loss(epoch_loss_dict, num_batch):
    for key in epoch_loss_dict:
            epoch_loss_dict[key] /= num_batch
    return epoch_loss_dict


def batch_report(loss, n_epoch, idx, num_batch, mode='training', verbose=False):
    if verbose:
        print(f'------------{mode}------------')
        print('Epoch: [{0}][{1}/{2}]'.format(n_epoch, idx, num_batch))
        print(f"\t total loss: {loss['loss']:.3f}")
        print(f"\t itc loss: {loss['loss_itc']:.3f}")
        print(f"\t itm loss: {loss['loss_itm']:.3f}")
        print(f"\t lm loss: {loss['loss_lm']:.3f}")


def scheduler_show(optimizer_scheduler, verbose=False):
    schedule_params = {}
    schedule_params['lr'] = optimizer_scheduler.optimizer.param_groups[0]['lr']
    if verbose:
        print(schedule_params)
    return schedule_params
    

def train(model, dataloader, optimizer_scheduler, writer_names, loss_writers, scheduler_writers, n_epoch, VERBOSE, scaler=None):
    model.train()
    epoch_loss_dic = init_loss_dic(writer_names)

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        #try:
        optimizer_scheduler.optimizer_zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = model(*batch)
            scaler.scale(loss['loss']).backward()
            scaler.step(optimizer_scheduler.optimizer)
            scaler.update()
            optimizer_scheduler.step(optimizer_step=False)
        else:
            loss = model(*batch)
            loss['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_scheduler.clip)
            optimizer_scheduler.step()

        epoch_loss_dic = accumulate_loss_dic(writer_names, epoch_loss_dic, loss)
        batch_loss_dic = write_loss_to_dic(writer_names, loss)
        train_step = n_epoch * len(dataloader) + idx
        if loss_writers is not None:
            loss_writers.write_task('train', batch_loss_dic, train_step)
            batch_report(batch_loss_dic, n_epoch, idx, len(dataloader), mode='train', verbose=VERBOSE)

        scheduler_dic = scheduler_show(optimizer_scheduler, verbose=VERBOSE)
        if scheduler_writers is not None:
            scheduler_writers.write_task('train', scheduler_dic, train_step)
    
    #model.module.clear_queues()    
    #except Exception as exc:
    #    print(exc)
    #    print(batch[0].shape, batch[1].shape)
    #    continue

    scheduler_show(optimizer_scheduler, verbose=True)
    epoch_loss_dic = average_epoch_loss(epoch_loss_dic, len(dataloader))
    return epoch_loss_dic


def val(model, dataloader, writer_names, summary_writers, n_epoch, VERBOSE):
    model.eval()
    epoch_loss_dic = init_loss_dic(writer_names)
    
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        #try:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                loss = model(*batch)#, **input_params)
        epoch_loss_dic = accumulate_loss_dic(writer_names, epoch_loss_dic, loss)
        batch_loss_dic = write_loss_to_dic(writer_names, loss)
        if summary_writers is not None:
            batch_report(batch_loss_dic, n_epoch, idx, len(dataloader), mode='validation', verbose=VERBOSE)

    #model.module.clear_queues()
    #except Exception as exc:
    #    print(exc)
    #    print(batch[0].shape, batch[1].shape)
    #    continue
    epoch_loss_dic = average_epoch_loss(epoch_loss_dic, len(dataloader))
    if summary_writers is not None:
        summary_writers.write_task('val', epoch_loss_dic, n_epoch)
    return epoch_loss_dic

def epoch_report(start_time, end_time, train_loss, valid_loss, n_epoch):
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {n_epoch + 1:02} | '
              f'Time: {epoch_mins}m {epoch_secs}s',
              flush=True)
        print(f'\tTrain Loss: {train_loss:.3f}', flush=True)
        print(f'\t Valid. Loss: {valid_loss:.3f}', flush=True)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']= '0, 1, 2, 3'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    MODEL_NAME = 'blip2_style_learning_WO_LMD_224notes_12bins'
    DEBUG = 0

    if DEBUG:
        save_root = 'blip2_style_learning/save'
        log_path_name = 'debug'
    else:
        save_root = '/data1/zhaojw/blip2_a2s/'
        log_path_name = MODEL_NAME


    readme_fn = 'blip2_style_learning/train_stage1.py'
    log_path_mng = LogPathManager(readme_fn, save_root=save_root, log_path_name=log_path_name)

    world_size = torch.cuda.device_count()
    #print(world_size)
    mp.spawn(main, args=(world_size, log_path_mng, DEBUG, MODEL_NAME), nprocs=world_size)
