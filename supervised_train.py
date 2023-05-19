# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import argparse
import os
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets
from pytorch_msssim import ms_ssim
# from torch.utils.tensorboard import SummaryWriter

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *


LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0

# msssim_weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]     # how to get msssim weight??   --zys
data_range = 1.0     #  how to get data range?    --zys

l1_criterion = nn.L1Loss()


def create_data_loaders(train_transformation,       # **dataset_config, args=args  --zys
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)


    train_loader = torch.utils.data.DataLoader(dataset,        # include labeled and unlabeled dataset  --zys
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader

'''I don't know what's this  --zys'''
def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


'''Training process of mean-teacher architecture  --zys'''
def train(train_loader, model, writer, optimizer_model, epoch, log):
    global global_step

    # for class criterion  --zys
    class_criterion = nn.CrossEntropyLoss().cuda()  # for class cost, between labeled target and student model  --zys

    # class_criterion = nn.CrossEntropyLoss().cuda()


    """Computes and stores the average and current value"""
    meters = AverageMeterSet()

    # switch to train mode
    model.train()

    end = time.time()
    print(optimizer_model.param_groups[0]['lr'])
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer_model, writer, epoch, i, len(train_loader))

        meters.update('lr', optimizer_model.param_groups[0]['lr'])


        input_var = input.to('cuda')    # --zys
        target_var = target.to('cuda')   # --zys
        # print(target_var.shape)

        labeled_minibatch_size = len(target_var)

        latent_dimension, class_logit = model.encoder(input_var)              # output of student model for labeled data  --zys
        


        '''generate z_hat and fake data'''
        # z_hat = model.alpha * model_latent + (1-model.alpha) * ema_latent
        z_hat = latent_dimension
        z_hat = z_hat.unsqueeze(-1).unsqueeze(-1)
        fake_data = model.decoder(z_hat)
        


        class_loss = class_criterion(class_logit, target_var) #   class_loss   --zys
        # class_loss = class_criterion(class_logit, target_var)
        meters.update('class_loss', class_loss.data)   # update: self.val = val, self.sum += val * n self.count += nself.avg = self.sum / self.count   --zys

        '''ms_ssim_l1 loss'''
        weight = 0.85
        ms_ssim_batch_wise = 1 - ms_ssim(F.interpolate(input_var, size=((32,32))), fake_data, data_range=data_range,
                                          size_average=True, win_size=1)
        l1_batch_wise = l1_criterion(F.interpolate(input_var, size=((32,32))), fake_data) / data_range
        ms_ssim_l1_loss = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise
        
        
        

        # get weight of ms_ssim_l1_loss  --zys
        # rampup_weight = get_current_consistency_weight(epoch)

        # loss = class_loss + consistency_loss + ms_ssim_l1_loss + res_loss   # if logit_distance_cost == -1, res_loss = 0.   --zys
        loss = class_loss + 0.05 * ms_ssim_l1_loss

        # update loss    --zys
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        
        meters.update('loss', loss.data)  #   --zys
        meters.update('ms_ssim_l1_loss', ms_ssim_l1_loss.data)    #   --zys


        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))    # top1 and top5 accuracy of stu model.   --zys
        meters.update('top1', prec1, labeled_minibatch_size)
        meters.update('error1', 100. - prec1, labeled_minibatch_size)
        meters.update('top5', prec5, labeled_minibatch_size)
        meters.update('error5', 100. - prec5, labeled_minibatch_size)



        global_step += 1   # theta = alpha*theta_{t-1} + (1-theta)*theta_t, 'ema-decay' means momentum  --zys
        writer.add_scalar('Training Class_Loss', class_loss, global_step=global_step)
        writer.add_scalar('Training ssim_l1_Loss', ms_ssim_l1_loss, global_step=global_step)
        writer.add_scalar('Training Loss', loss, global_step=global_step)
        

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time: {meters[batch_time]:.3f}\t'
                'Data: {meters[data_time]:.3f}\t'
                'Class Loss: {meters[class_loss]:.5f}\t'
                'ms_ssim_l1_loss Loss: {meters[ms_ssim_l1_loss]:.5f}\t'
                'Prec@1: {meters[top1]:.2f}%\t'
                'Prec@5: {meters[top5]:.2f}%'.format(epoch, i, len(train_loader), meters=meters)
            )

            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })


def validate(eval_loader, model, writer, log, global_step, epoch):
    class_criterion = nn.CrossEntropyLoss().cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = input.to('cuda')
        # target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)
        target_var = target.to('cuda')     # --zys

        labeled_minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()

        # compute output
        latent_dimension, class_logit = model.encoder(input_var)

        class_loss = class_criterion(class_logit, target_var)
        

        # measure accuracy and record loss
        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        # print(prec1, prec5)
        meters.update('class_loss', class_loss.data, labeled_minibatch_size)
        meters.update('top1', prec1, labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1, labeled_minibatch_size)
        meters.update('top5', prec5, labeled_minibatch_size)
        meters.update('error5', 100.0 - prec5, labeled_minibatch_size)
        
        writer.add_scalar('Validate Loss', class_loss, global_step=epoch)
        writer.add_scalar('top1', prec1, global_step=epoch)


        # writer.add_scalar('Training Loss', prec1, global_step=global_step)
        # writer.add_scalar('Training Accuracy', prec5, global_step=global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Time: {meters[batch_time]:.3f}\t'
                'Data: {meters[data_time]:.3f}\t'
                'Class Loss: {meters[class_loss]:.5f}\t'
                'Prec@1: {meters[top1]:2f}%\t'
                'Prec@5: {meters[top5]:2f}%'.format(
                    i, len(eval_loader), meters=meters))

    # LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
    #       .format(top1=meters['top1'], top5=meters['top5']))
    log.record(epoch, {
        'step': global_step,
        **meters.values(),
        **meters.averages(),
        **meters.sums()
    })

    return meters['top1'].avg


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer_model, writer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer_model.param_groups:
        param_group['lr'] = lr
        
    writer.add_scalar('Learning Rate', lr, global_step=epoch)






def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # if k == 5:
        #     print(correct[:k].view(-1))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        # print(correct_k.mul_(100.0 / labeled_minibatch_size).shape)   #  --  zys
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
    return res

def main(context):
    global global_step
    global best_prec1

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    summary_log = SummaryWriter(context.summary)
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    dataset_config = datasets.__dict__[args.dataset]()   # dictionary type, return train_transformation, envl _trans, dataset dir, num_classes...   --zys
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)        # get dataloader, trainloader include labeled and unlabeled dataset --zys

    def create_model(ema=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))



        # model_factory = architectures.__dict__[args.arch]  # args.arch = cifar_shakeshake26  --zys
        # model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
        # model = model_factory(**model_params)
        # model = nn.DataParallel(model).cuda()  # If you have only one GPU, you don't need it, but can still use  --zys

        model_params = dict(name='mobilenet_v3_small', head='linear', num_classes=num_classes, batch_size=args.batch_size)
        model = architectures.sup_model(**model_params)
        # SimCLR = torch.load('SimCLR.pth')
        # model.load_state_dict(SimCLR['model'], strict=False)


        '''Override create model'''


        if ema:
            for param in model.parameters():
                param.detach_()  # the results will never require gradient, just need calculate mean and update by EMA method. --zys
            # model.head_alpha.requires_grad = False    #  --zys

        return model

    '''Create two different model here  --zys'''
    model = create_model().to('cuda')   # this is student model   --zys

    '''Construct decoder model    --zys'''
    # generator = architectures.Generator().to('cuda')


    LOG.info(parameters_string(model))

    '''Create SGD optimizer, we can consider using ADAM  --zys'''
    optimizer_model = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    # optimizer_model = torch.optim.Adam(model.parameters(), lr=args.initial_lr, betas=(0.95, 0.95), weight_decay=1e-6)

    # optimizer_ema = torch.optim.SGD(ema_model.parameters(), args.lr,
    #                                   momentum=args.momentum,
    #                                   weight_decay=args.weight_decay,
    #                                   nesterov=args.nesterov)

    # optimizer_g = torch.optim.SGD(generator.parameters(), args.lr,
    #                                 momentum=args.momentum,
    #                                 weight_decay=args.weight_decay,
    #                                 nesterov=args.nesterov)
    # optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)


    # optionally resume from a checkpoint, call load_state_dict() from pytorch.   --zys
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        # optimizer_model.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        
        # import torch
        # checkpoint = torch.load('./results/main_global/2022-03-30_17:28:17/0/checkpoint/best.ckpt')
        # checkpoint['best_prec1']
        

    cudnn.benchmark = True

    '''validate pretrained model   --zys'''
    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model, summary_log, validation_log, global_step, args.start_epoch)
        LOG.info("Evaluating the EMA model:")
        validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch)
        return

    '''Process of training   --zys'''
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, summary_log, optimizer_model, epoch, training_log)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        '''evaluate every args.evaluation_epochs    --zys'''
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(eval_loader, model, summary_log, validation_log, global_step, epoch + 1)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            # flag best one of this epoch  --zys
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
        else:
            is_best = False

        '''save checkpoint every args.checkpoint_epochs    --zys'''
        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer_model.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)
        
        
            
    summary_log.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()     # 'cli.py' is config file
    main(RunContext(__file__, 0))
