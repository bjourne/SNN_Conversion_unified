import time
import numpy as np
import torch
from tqdm import tqdm
from utils import reset_net, regular_set

import random
import os

from misc import accuracy, save_checkpoint, AverageMeter, ProgressMeter
from logger import Logger
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

DEVICE = "cpu"

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def eval_ann(test_dataloader, model, crit, device):
    model.eval()
    # model.to(device)
    tot = torch.tensor(0.).to(device)  # accuracy
    length = 0
    epoch_loss = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = crit(out, label)
            epoch_loss += loss.item()
            length += len(label)
            tot += (label==out.max(1)[1]).sum().data
    return tot/length, epoch_loss/length


def eval_snn(test_dataloader, model, device, sim_len=8):
    tot = torch.zeros(sim_len).to(device)
    length = 0
    model.eval()
    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(test_dataloader)):
            spikes = 0
            length += len(label)
            img = img.to(device)
            label = label.to(device)
            for t in range(sim_len):
                out = model(img)
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum()
            reset_net(model)
    return tot/length


def train(loader, net, crit, optimizer, epoch, args):
    batch_time = AverageMeter(name='Time', fmt=':6.3f')
    losses = AverageMeter(name='Loss', fmt=':6.3f')  # fmt=':.4e' with 1.1337e+00
    top1 = AverageMeter(name='Acc@1', fmt=':6.2f')
    top5 = AverageMeter(name='Acc@5', fmt=':6.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    # ## Use to calculate the loss
    running_loss = 0
    n_correct = 0
    n_total = 0

    net.train()
    end = time.time()
    for i, (images, yreal) in enumerate(loader):
        n_batch = images.size(0)
        images = images.to(DEVICE, non_blocking=True)
        yreal = yreal.to(DEVICE, non_blocking=True)

        yhat = net(images)


        loss = crit(yhat, yreal)
        acc1, acc5 = accuracy(yhat, yreal, topk=(1, 5))
        losses.update(loss.item(), n_batch)
        top1.update(acc1.item(), n_batch)
        top5.update(acc5.item(), n_batch)

        running_loss += loss.item()
        _, predicted = yhat.max(dim=1)

        batch_correct = predicted.eq(yreal).sum().item()
        n_correct += batch_correct
        n_total += n_batch
        if torch.isnan(torch.tensor(loss.item())):
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if (i % args.print_freq == 0) or (i == len(loader)-1):
            progress.display(i + 1)

    avg_loss = running_loss / (i+1)
    accu = n_correct / n_total
    # ## Print train/test loss/accuracy
    print(f"Train AvgLoss: {avg_loss:>.3f}, Accuracy: {(100*accu):>0.2f}% .")
    print(f"Train loss.avg: {losses.sum /losses.count:>0.3f}, "
          f"Top 1 avg: {top1.avg:>.3f}%, Top 5 avg: {top5.avg :>.3f}% .")
    # losses.avg  == losses.sum / losses.count
    print("==> Display training information")
    progress.display_summary()
    return (losses.avg, top1.avg, top5.avg)


def test(test_loader, model, crit, args):
    batch_time = AverageMeter(name='Time', fmt=':6.3f')
    losses = AverageMeter(name='Loss', fmt=':6.3f')
    top1 = AverageMeter(name='Acc@1', fmt=':6.2f')
    top5 = AverageMeter(name='Acc@5', fmt=':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # ## Use to calculate the loss
    running_loss = 0
    n_correct = 0
    n_total = 0  # len(trainloader.dataset) # which is num_total
    # ## Switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, yreal) in enumerate(test_loader):
            # ## Move data to the same device as model
            images = images.to(DEVICE, non_blocking=True)
            yreal = yreal.to(DEVICE, non_blocking=True)
            # ## Compute output
            output = model(images)
            # ## Measure accuracy and record loss
            loss = crit(output, yreal)
            acc1, acc5 = accuracy(output, yreal, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            running_loss += loss.item()
            _, predicted = output.max(dim=1)

            batch_correct = predicted.eq(yreal).sum().item()
            n_correct += batch_correct
            n_total += yreal.size(0)
            # ## Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i % args.print_freq == 0) or (i == len(test_loader)-1):
                progress.display(i + 1)

    avg_loss = running_loss / (i+1)
    accu = n_correct / n_total
    # ## Print train/test loss/accuracy
    print(f"Test AvgLoss: {avg_loss:>.3f}, Accuracy: {(100*accu):>0.2f}% .")
    print(f"Test loss.avg: {losses.sum /losses.count:>0.3f}, "
          f"Top 1 avg: {top1.avg:>.3f}%, Top 5 avg: {top5.avg :>.3f}% .")
    print("==> Display testing information")
    progress.display_summary()
    return (losses.avg, top1.avg, top5.avg)



def train_ann_flag(
    train_loader,
    test_loader,
    model,
    crit,
    optimizer,
    scheduler,
    args,
    state
):
    model.to(DEVICE)
    # ## model.train()
    logger = Logger(os.path.join(args.checkpoint, args.dataset, args.name + '.txt'), title='train_ann')
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Train Acc1.', 'Train Acc5.', 'Test Loss', 'Test Acc1.', 'Test Acc5.'])
    logger.set_formats(['{0:d}', '{0:.7f}', '{0:.4f}', '{0:.3f}', '{0:.3f}', '{0:.4f}', '{0:.3f}', '{0:.3f}'])
    best_acc1 = 0
    is_best = 0
    mom_disflag = 0
    for epoch in range(args.epochs):
        state['lr'] = optimizer.state_dict()['param_groups'][0]['lr']
        print('Epoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))

        loss_train, acc1_train, acc5_train = train(train_loader, model, crit, optimizer, epoch, args)
        if torch.isnan(torch.tensor(loss_train)):
            mom_disflag = 1
            break

        loss_test, acc1_test, acc5_test = test(test_loader, model, crit, args)
        print('Epoch {} --> Val_loss: {}, Acc: {}'.format(epoch, loss_test, acc1_test), flush=True)
        # ## Append logger file
        logger.append([epoch, state['lr'], loss_train, acc1_train, acc5_train, loss_test, acc1_test, acc5_test])
        # ## Remember best acc@1 and save checkpoint
        is_best = acc1_test > best_acc1
        best_acc1 = max(acc1_test, best_acc1)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'model_arch': args.model,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                },
            is_best,
            checkpoint=args.checkpoint+'/'+args.dataset,
            filename=args.name)
        # ## Updating the learning rate for the next epoch
        scheduler.step()
    # ## Finish write information
    logger.close()
    return mom_disflag, best_acc1, model, logger


def train_ann_(train_loader, test_loader, model, crit, args, state):
    model.to(DEVICE)
    # ## SGD with momentum
    # ## If use all parameter, worse performance.
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    # ## SGD without momentum
    optimizer1 = SGD(model.parameters(), lr=args.lr)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=args.epochs )
    print('============= Try to train the model with SGD with MOMENTUM ============= ')
    mom_disflag, best_acc1, model, logger = train_ann_flag(
        train_loader, test_loader, model, crit, optimizer, scheduler, args, state)
    if mom_disflag == 1:
        print('================ Failed using SGD with momentum, and try SGD without momentum ================')
        mom_disflag, best_acc1, model, logger = train_ann_flag(
            train_loader, test_loader, model, crit, optimizer1, scheduler1, args, state)
        print('================ Failed using SGD without momentum, and stop ================')
    return best_acc1, model, logger



def train_ann(
    train_loader,
    test_loader,
    net,
    crit,
    args,
    state
):
    """ Imported from QCFS, Only part of the model parameters are in the optimizer """
    net.to(DEVICE)
    # ## SGD with momentum
    para1, para2, para3 = regular_set(net)
    optimizer = SGD(
        [
            {'params': para1, 'weight_decay': args.weight_decay},
            {'params': para2, 'weight_decay': args.weight_decay},
            {'params': para3, 'weight_decay': args.weight_decay}
            ],
        lr=args.lr,
        momentum=args.momentum
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    # ## SGD without momentum
    opt1 = torch.optim.SGD(
        [
            {'params': para1, 'weight_decay': args.weight_decay},
            {'params': para2, 'weight_decay': args.weight_decay},
            {'params': para3, 'weight_decay': args.weight_decay}
            ],
        lr=args.lr)

    scheduler1 = CosineAnnealingLR(opt1, T_max=args.epochs )
    mom_disflag, best_acc1, net, logger = train_ann_flag(
        train_loader, test_loader, net, crit, optimizer, scheduler, args, state)
    # ## The first try may change a bit of the model.
    if mom_disflag == 1:
        print('================ Failed using SGD with momentum, and try SGD without momentum ================')
        mom_disflag, best_acc1, net, logger = train_ann_flag(
            train_loader, test_loader, net, crit, opt1, scheduler1, args, state)
        print('================ Failed using SGD without momentum, and stop ================')
    return best_acc1, net, logger
