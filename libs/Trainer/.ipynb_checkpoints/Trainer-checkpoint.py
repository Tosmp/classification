import time
import torch
from torch.cuda import amp
import sys
sys.path.append("./libs/utils")
from eval import AverageMeter, accuracy
from torch.utils.tensorboard import SummaryWriter


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
    
class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion.to(device)
        self.scaler = amp.GradScaler() 
        self.device = device
        self.config = config
        self.top1 = AverageMeter()
        self.t_losses = AverageMeter()
        self.v_losses = AverageMeter()
        self.writer = SummaryWriter(log_dir="./logs/"+self.config.f_name)
        self.iter = 0
        
    def fit(self, train_dl, val_dl):
        torch.backends.cudnn.benchmark = True
        best_top1 = 0.0
        for i in range(self.config.n_epochs):
            start = time.time()
            self.train_one_epoch(train_dl)
            self.validation(val_dl)
            lr = self.optimizer.param_groups[0]['lr']
            self.log(f'[RESULT]: Epoch: {i+1}, train_loss: {self.t_losses.avg:.4f}, val_loss: {self.v_losses.avg:.5f}, top1: {self.top1.avg:.3f}, lr: {lr:.5f}, time: {(time.time() - start):.3f}')
            self.writer.add_scalar('train_loss', round(self.t_losses.avg, 5), i+1)
            self.writer.add_scalar('val_loss', round(self.v_losses.avg, 5), i+1)
            self.writer.add_scalar('top1_acc', round(self.top1.avg, 5), i+1)
            
            if best_top1 < self.top1.avg:
                best_top1 = self.top1.avg
                self.save(epoch=i+1, top1=self.top1.avg)
        self.save(epoch=self.config.n_epochs, top1=self.top1.avg, last=True)
        self.writer.close()
    def train_one_epoch(self, train_dl):
        self.model.train()
        self.t_losses.reset()
        itr_start = time.time()
        for img, target in train_dl:
            self.optimizer.zero_grad()
            with amp.autocast(enabled=True):
                pred = self.model(img.to(self.device))
                loss = self.criterion(pred, target.to(self.device))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.t_losses.update(loss.item(), self.config.train_batch_size)
            
            #------------可視化------------
            if (self.iter % 100) == 0 :
                print(f'iter : {self.iter} time : {int(time.time() - itr_start)}sec')
                itr_start = time.time()
            self.iter += 1
    def validation(self, val_dl):
        self.model.eval()
        self.top1.reset()
        self.v_losses.reset()
        for img, target in val_dl:
            with amp.autocast(enabled=True):
                pred = self.model(img.to(self.device))
                loss = self.criterion(pred, target.to(self.device))
            [prec1] = accuracy(pred.data.cpu().float(), target.cpu().float(), topk=(1, ))
            self.top1.update(to_python_float(prec1), img.size(0))
            self.v_losses.update(loss.item(), self.config.val_batch_size)
            
    def save(self, epoch, top1, last=False):
        if last:
            l_or_b = '_last.bin'
        else:
            l_or_b = '_best.bin'
        torch.save({
            'model_state_dict': self.model.state_dict(), 
            'top1': top1,
            'epoch': epoch,
        }, self.config.weight_path + l_or_b)
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.config.log_path + '.txt', mode='a') as logger:
            logger.write(f'{message}\n')