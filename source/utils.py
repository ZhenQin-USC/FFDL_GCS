import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from os.path import join
from collections import OrderedDict
import h5py
from torch.utils.data import Dataset #, DataLoader
from tqdm import tqdm
import os
import json
from torch.utils.data import Dataset, DataLoader
import psutil


class RSE_loss(object):
    def __init__(self, p=2, d=None, split=False, weights=None):
        super(RSE_loss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert p > 0
        self.d = d
        self.p = p
        if weights == None:
            weights = [1, 1]
        self.p_weight, self.s_weight = weights[0], weights[1]
        self.split = split

    def rel_square_error(self, y_pred, y):
        diff_norms = torch.norm(y_pred-y, p=self.p, dim=self.d)
        y_norms = torch.norm(y, p=self.p, dim=self.d)
        return torch.mean(diff_norms/y_norms)

    def __call__(self, y_pred, y):
        if self.split:
            p_pred, s_pred = torch.split(y_pred, (1, 1), dim=2)
            p_true, s_true = torch.split(y, (1, 1), dim=2)
            loss = self.p_weight*self.rel_square_error(p_pred, p_true)
            loss += self.s_weight*self.rel_square_error(s_pred, s_true)
        else:
            loss = self.rel_square_error(y_pred, y)
        return loss


class MyDataset(Dataset):
    def __init__(self, data_loader, nstep:int, nt:int=8, stride:int=1, interval:int=1, fixed_control:bool=False):

        self.contrl, self.states, self.static = data_loader()
        nsample = self.states.shape[0]
        time_index, real_index = get_slice_index(
            nsample, nstep, nt=nt, stride=stride, interval=interval)
        self.time_index = time_index
        self.real_index = real_index
        self.stride = stride
        self.fixed_control = fixed_control
        
    def __getitem__(self, index):
        # contrl -- (nsample, nstep, 2, nx, ny, nz) 
        # states -- (nsample, nstep+1, 2, nx, ny, nz) 
        # static -- (nsample, nstatic, nx, ny, nz)
        _real_index = self.real_index[index]
        _time_index = self.time_index[index]
        if self.fixed_control:  # fixed_control = True
            contrl = self.contrl[0, _time_index, ...]
        else: # fixed_control = False
            contrl = self.contrl[_real_index, _time_index, ...]
        states = self.states[_real_index, _time_index[0], ...]
        static = self.static[_real_index, ...]
        output = self.states[_real_index, _time_index+self.stride, ...]
        return ((contrl, states, static), output)
    
    def __len__(self):
        return len(self.time_index)


def memory_usage_psutil():
    # return the memory usage in percentage like top
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss/(1e3)**3
    print('Memory Usage in Gb: {:.2f}'.format(mem))  # in GB 
    return mem


def get_slice_index(nsample:int, nstep:int, nt:int=8, stride:int=1, interval:int=1):
    kk = 0
    time_index = []
    for step in np.array(range(0, nstep-nt+1, stride)):
        index = np.array(range(step, step+nt*interval, interval))
        if index[-1] <= nstep-1:
            # print(step, index, index+1, index[-1])
            time_index.append(index)
            kk += 1
            
    time_index = np.repeat(np.array(time_index)[None], repeats=nsample, axis=0)
    # time_index = merge_first_two_dim(time_index)
    time_index = np.vstack(time_index)

    real_index = np.array(range(nsample)).repeat(kk, axis=0)

    return time_index, real_index


def load_data_with_index(path_to_data, sample_index, nstep=16, p_postprocess=None):
    s_all, p_all, u_all, k_all = [], [], [], []

    for Ii in sample_index:
        s = torch.load(join(path_to_data, 'processed_plume_{}.pt'.format(Ii+1)))
        p = torch.load(join(path_to_data, 'processed_pressure_{}.pt'.format(Ii+1)))
        u = torch.load(join(path_to_data, 'processed_rate_{}.pt'.format(Ii+1)))
        k = torch.load(join(path_to_data, 'processed_static_{}.pt'.format(Ii+1)))
        s_all.append(s)
        p_all.append(p)
        u_all.append(u)
        k_all.append(k)
        del s, p, u, k
        
    s = torch.vstack(s_all)[:, :nstep, None]
    p = torch.vstack(p_all)[:, :nstep, None]
    un= torch.vstack(u_all)[:,:nstep-1,None]
    static = torch.vstack(k_all)
    # print(s.shape, p.shape, un.shape, static.shape)
    del s_all, p_all, u_all, k_all

    # States:
    if p_postprocess != None:
        p  = p_postprocess(p)
    states = torch.cat((p, s), dim=2)
    del p, s
    
    # Control:
    uw = torch.zeros_like(un, dtype=torch.float32)
    contrl = torch.cat((uw, un), dim=2)
    del uw, un
    
    return contrl, states, static


def data_loader_experiment_1(folders, sample_index=range(10), nstep=16, root_to_data=None):
    p_postprocess = lambda x: (x-0.95)/3
    contrl, states, static = [], [], []
    if root_to_data == None:
        root_to_data = '/scratch1/zhenq/data/grid120'

    for folder in folders:
        mem = memory_usage_psutil()
        path_to_data = join(root_to_data, folder)
        _contrl, _states, _static = load_data_with_index(
            path_to_data, sample_index, nstep=nstep, p_postprocess=p_postprocess)
        print("Folder: ", folder, 
              "\nSample Index: ", sample_index, 
              "\nShapes of Loaded Samples: ", _contrl.shape, _states.shape, _static.shape)
        contrl.append(_contrl)
        states.append(_states)
        static.append(_static)
        del _contrl, _states, _static

    contrl = torch.vstack(contrl) #.shape
    states = torch.vstack(states)
    static = torch.vstack(static)

    return contrl, states, static


def extract_data_instance(data, device):

    inputs, outputs = data
    _contrl, _states, _static = inputs
    _contrl = _contrl.to(device)
    _states = _states.to(device)
    _static = _static.to(device)
    outputs = outputs.to(device)

    return _contrl, _states, _static, outputs


def train_one_epoch(model, device, train_loader, epoch, NUM_EPOCHS, 
                    optimizer, scheduler, loss_fn, verbose=0, 
                    gradient_clip=False, gradient_clip_val=None):
    batch_loss = 0.
    # last_loss = 0.

    if verbose == 1:
        loop = tqdm(train_loader)   
    elif verbose == 0:
        loop = train_loader

    for i, data in enumerate(loop):
        # Data instance 
        _contrl, _states, _static, outputs = extract_data_instance(data, device)

        # Zero gradients for every batch
        optimizer.zero_grad()

        # Make predictions for every batch
        preds, _ = model(_contrl, _states, _static)

        # Compute the loss and its gradients
        loss = loss_fn(preds, outputs)
        loss.backward()

        # Gradient Clip
        if gradient_clip:
            if gradient_clip_val==None:
                pass
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

        # Updates the parameters
        optimizer.step()

        # Adjust learning weights
        scheduler.step()

        # Gather data and report
        batch_loss += loss.item()

        if verbose == 1:
            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item()) 
        elif verbose == 0:
            pass

    return batch_loss/(i+1), optimizer, scheduler


def validate_model(model, device, valid_loader, loss_fn, metrics):

    model.eval()
    running_vloss = 0.0
    i = 0
    running_metrics = []
    for i, vdata in enumerate(valid_loader):

        _metrics_val = []
        _contrl, _states, _static, voutputs = extract_data_instance(vdata, device)
        
        with torch.no_grad():
            vpreds, _ = model(_contrl, _states, _static)
            vloss = loss_fn(vpreds, voutputs)
            
            for _metrics in metrics:
                metric = _metrics(vpreds, voutputs)
                _metrics_val.append(metric.item())

            running_metrics.append(_metrics_val)

        running_vloss += vloss.item() #float(vloss)
    
    return running_vloss/(i+1), list(np.array(running_metrics).mean(axis=0))


def train(model, device, EPOCHS, train_loader, valid_loader, path_to_model, 
          learning_rate=1e-3, step_size=250, gamma=0.975, verbose=0, loss_fn = nn.MSELoss(),
          gradient_clip=False, gradient_clip_val=None, if_track_validate=True, weight_decay=0):
    
    metrics = [RSE_loss(p=1), RSE_loss(p=2), nn.MSELoss()]
    metrics_name = ['L1_RSE', 'L2_RSE', 'MSE']

    train_loss_list = []
    valid_loss_list = []
    learning_rate_list = []
    metrics_val = []

    epoch_number = 0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=False)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Training 
        model.train(True)
        batch_loss, optimizer, scheduler = train_one_epoch(
            model, device, train_loader, epoch, EPOCHS, optimizer, scheduler, loss_fn, 
            verbose=verbose, gradient_clip=gradient_clip, gradient_clip_val=gradient_clip_val
            )
        train_loss_list.append(batch_loss)
        if if_track_validate == True:
            # Validation
            batch_vloss, _metrics_val = validate_model(model, device, valid_loader, loss_fn, metrics)
        else:
            batch_vloss = 0.0
            _metrics_val = 0.0
            
        valid_loss_list.append(batch_vloss)
        metrics_val.append(_metrics_val)
        print('Epoch {}: Loss train {} valid {}. LR {}'.
              format(epoch, batch_loss, batch_vloss, scheduler.get_last_lr()))
        print(_metrics_val)
        learning_rate_list.append(float(optimizer.param_groups[0]['lr']))
        
        # Save checkpoint every epoch
        if (epoch+1)%5 == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
        
        print('save model!!!') 
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, join(path_to_model, 'checkpoint{}.pt'.format(epoch_number)))
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, join(path_to_model, 'checkpoint.pt'))

        np.savez(join(path_to_model,'loss_epoch.npz'), 
                 train_loss=np.array(train_loss_list), 
                 valid_loss=np.array(valid_loss_list), 
                 metrics=np.array(metrics_val),
                 learning_rate=np.array(learning_rate_list)
                 )
        
        epoch_number += 1

    return train_loss_list, valid_loss_list


def test(model, data_loader, device, path_to_saved_data, metrics, use_cuda=True):
    mse_loss_fn, rse_loss_fn, l1_rse_loss_fn = metrics
    # data_loader = DataLoader(dataset, sampler=sampler)
    model.eval()
    if use_cuda:
        model.to(device)
    else:
        model.to('cpu')
    preds, trues = [], []
    mse_losses, rse_losses = [], []
    p_mse_losses, s_mse_losses = [], []
    p_rse_losses, s_rse_losses = [], []
    p_l1rse_losses, s_l1rse_losses = [], []
    loop = tqdm(data_loader)
    for data in loop:
        if use_cuda:
            _contrl, _states, _static, outputs = extract_data_instance(
                data, device)
        else:
            _contrl, _states, _static, outputs = extract_data_instance(data, 'cpu')

        with torch.no_grad():
            pred, _ = model(_contrl, _states, _static)
            _mse_loss = mse_loss_fn(pred, outputs).to('cpu').item()
            _rse_loss = rse_loss_fn(pred, outputs).to('cpu').item()

        trues.append(outputs.to('cpu').detach().numpy())
        preds.append(pred.to('cpu').detach().numpy())
        _mse_loss = mse_loss_fn(pred, outputs).to('cpu').item()
        _rse_loss = rse_loss_fn(pred, outputs).to('cpu').item()
        _p_mse_losses = mse_loss_fn(pred[:, :, 0], outputs[:, :, 0]).to('cpu').item()
        _s_mse_losses = mse_loss_fn(pred[:, :, 1], outputs[:, :, 1]).to('cpu').item()
        _p_rse_losses = rse_loss_fn(pred[:, :, 0], outputs[:, :, 0]).to('cpu').item()
        _s_rse_losses = rse_loss_fn(pred[:, :, 1], outputs[:, :, 1]).to('cpu').item()
        _p_l1rse_losses = l1_rse_loss_fn(pred[:, :, 0], outputs[:, :, 0]).to('cpu').item()
        _s_l1rse_losses = l1_rse_loss_fn(pred[:, :, 1], outputs[:, :, 1]).to('cpu').item()
        del _contrl, _states, _static, outputs, pred, _  # , state0

        # print(_mse_loss, _rse_loss)
        mse_losses.append(_mse_loss)
        rse_losses.append(_rse_loss)
        p_mse_losses.append(_p_mse_losses)
        s_mse_losses.append(_s_mse_losses)
        p_rse_losses.append(_p_rse_losses)
        s_rse_losses.append(_s_rse_losses)
        p_l1rse_losses.append(_p_l1rse_losses)
        s_l1rse_losses.append(_s_l1rse_losses)
        loop.set_postfix(mse=_mse_loss, rse=_rse_loss,
                         p_mse=_p_mse_losses, s_mse=_s_mse_losses,
                         p_rse=_p_rse_losses, s_rse=_s_rse_losses)
    # Collect Testing Results
    mse_losses = np.array(mse_losses)
    rse_losses = np.array(rse_losses)
    p_mse_losses = np.array(p_mse_losses)
    s_mse_losses = np.array(s_mse_losses)
    p_rse_losses = np.array(p_rse_losses)
    s_rse_losses = np.array(s_rse_losses)
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    p_pred, s_pred = preds[:, :, 0], preds[:, :, 1]
    p_true, s_true = trues[:, :, 0], trues[:, :, 1]
    print(mse_losses.shape, rse_losses.shape)
    print(p_pred.shape, s_pred.shape, p_true.shape, s_true.shape)
    print(preds.shape, trues.shape)
    np.savez(path_to_saved_data,
             mse_loss=mse_losses,
             rse_lose=rse_losses,
             p_mse_losses=p_mse_losses,
             s_mse_losses=s_mse_losses,
             p_rse_losses=p_rse_losses,
             s_rse_losses=s_rse_losses,
             preds=preds,
             trues=trues)
    # matdict = {'preds': preds, 'trues': trues}
    # sio.savemat(join(path_to_model, '{}_test_result.mat'.format(folder)), matdict)
    del preds, trues, data_loader

    return p_pred, s_pred, p_true, s_true

