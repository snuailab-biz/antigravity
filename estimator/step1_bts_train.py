# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import time
import argparse

import wandb

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.cm
from tqdm import tqdm

from bts_dataloader import *


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.')

parser.add_argument('--mode',                      type=str,   default='train')
parser.add_argument('--model_name',                type=str,   default='anti_densenet161_pad_only_100')
parser.add_argument('--encoder',                   type=str,   default='densenet161_bts')

# Dataset
parser.add_argument('--dataset',                   type=str,   default='azure_kinect_pad')
parser.add_argument('--seg_path',                  type=str,   default='E:/bts-master/pytorch/dataset/dataset_bts_train/segmentations')
parser.add_argument('--data_path',                 type=str,   default='E:/bts-master/pytorch/dataset/dataset_bts_train/images')
parser.add_argument('--gt_path',                   type=str,   default='E:/bts-master/pytorch/dataset/dataset_bts_train/depths')
parser.add_argument('--filenames_file',            type=str,   default='E:/bts-master/pytorch/dataset/dataset_bts_train/train_file_list.txt')

# image default ratio : 16:9 -> 32multiple -> 512*288
parser.add_argument('--input_width',               type=int,   default=512)
parser.add_argument('--input_height',              type=int,   default=288)
parser.add_argument('--max_depth',                 type=float, default=10)

# Log and save
parser.add_argument('--wandb_enable',              type=bool,  default=False)
parser.add_argument('--log_directory',             type=str,   default='./models')
parser.add_argument('--checkpoint_path',           type=str,   default='')
parser.add_argument('--log_freq',                  type=int,   default=100)
parser.add_argument('--save_freq',                 type=int,   default=500)

# Training
parser.add_argument('--fix_first_conv_blocks',                 action='store_true')
parser.add_argument('--fix_first_conv_block',                  action='store_true')
parser.add_argument('--bn_no_track_stats',                     action='store_true')
parser.add_argument('--weight_decay',              type=float, default=1e-2)
parser.add_argument('--bts_size',                  type=int,   default=512)
parser.add_argument('--retrain',                               default=True)
parser.add_argument('--adam_eps',                  type=float, default=1e-3)
parser.add_argument('--batch_size',                type=int,   default=4)
parser.add_argument('--num_epochs',                type=int,   default=400)
parser.add_argument('--learning_rate',             type=float, default=1.414e-4)
parser.add_argument('--end_learning_rate',         type=float, default=-1)
parser.add_argument('--variance_focus',            type=float, default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',                      default=True)
parser.add_argument('--degree',                    type=float, default=1.0)
parser.add_argument('--do_kb_crop',                            default=True)
parser.add_argument('--use_right',                             action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   default=1)
parser.add_argument('--world_size',                type=int,   default=1)
parser.add_argument('--rank',                      type=int,   default=0)
parser.add_argument('--dist_url',                  type=str,   default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   default='nccl')
parser.add_argument('--gpu',                       type=int,   default=None)
parser.add_argument('--multiprocessing_distributed',           action='store_true')
# Online eval
parser.add_argument('--do_online_eval',                        action='store_true')
parser.add_argument('--data_path_eval',            type=str,   required=False)
parser.add_argument('--gt_path_eval',              type=str,   required=False)
parser.add_argument('--filenames_file_eval',       type=str,   required=False)
parser.add_argument('--min_depth_eval',            type=float, default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, default=80)
parser.add_argument('--eigen_crop',                            action='store_true')
parser.add_argument('--garg_crop',                             action='store_true')
parser.add_argument('--eval_freq',                 type=int,   default=500)
parser.add_argument('--eval_summary_directory',    type=str,   default='')


args = parser.parse_args()

if args.mode == 'train' and not args.checkpoint_path:
    from bts import *

elif args.mode == 'train' and args.checkpoint_path:
    model_dir = os.path.dirname(args.checkpoint_path)
    model_name = os.path.basename(model_dir)
    import sys
    sys.path.append(model_dir)
    for key, val in vars(__import__(model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


def set_misc(model):
    if args.bn_no_track_stats:
        print("Disabling tracking running stats in batch norm layers")
        model.apply(bn_init_as_tf)

    if args.fix_first_conv_blocks:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', 'base_model.layer1.1', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'denseblock1.denselayer2', 'norm']
        print("Fixing first two conv blocks")
    elif args.fix_first_conv_block:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
        print("Fixing first conv block")
    else:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', '.bn']
        else:
            fixing_layers = ['conv0', 'norm']
        print("Fixing first conv layer")

    for name, child in model.named_children():
        if not 'encoder' in name:
            continue
        for name2, parameters in child.named_parameters():
            # print(name, name2)
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False


def online_eval(model, dataloader_eval, gpu, ngpus):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depths']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depths. continue.')
                continue

            _, _, _, _, pred_depth = model(image, focal)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                else:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval test_samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.3f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Create model
    model = BtsModel(args)
    model.train()
    model.decoder.apply(weights_init_xavier)
    set_misc(model)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Training parameters
    optimizer = torch.optim.AdamW([{'params': model.module.encoder.parameters(), 'weight_decay': args.weight_decay},
                                   {'params': model.module.decoder.parameters(), 'weight_decay': 0}],
                                  lr=args.learning_rate, eps=args.adam_eps)

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                best_eval_steps = checkpoint['best_eval_steps']
            except KeyError:
                print("Could not load values for online evaluation")

            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True

    if args.retrain:
        global_step = 0

    cudnn.benchmark = True

    dataloader = BtsDataLoader(args, 'train')
    dataloader_eval = BtsDataLoader(args, 'online_eval')

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)

        if args.wandb_enable:
            wandb.init(project="BTS", entity="hanna")
            wandb.config = {
                "learning_rate": args.learning_rate,
                "encoder": args.encoder,
                "epochs": args.num_epochs,
                "batch_size": args.batch_size
            }

        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum = [var.sum().cpu().detach().numpy() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):

            optimizer.zero_grad()
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            focal = torch.autograd.Variable(sample_batched['focal'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depths'].cuda(args.gpu, non_blocking=True))

            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)

            if args.dataset == 'nyu':
                mask = depth_gt > 0.1

            elif args.dataset == 'azure_kinect_pad':
                mask = depth_gt > 0

            else:
                mask = depth_gt > 1.0

            loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum().cpu().detach().numpy() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('silog_loss', loss, global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)
                    depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                    for i in range(num_log_images):
                        writer.add_image('depth_gt/image/{}'.format(i), normalize_result(1/depth_gt[i, :, :, :].data), global_step)
                        writer.add_image('depth_est/image/{}'.format(i), normalize_result(1/depth_est[i, :, :, :].data), global_step)
                        writer.add_image('reduc1x1/image/{}'.format(i), normalize_result(1/reduc1x1[i, :, :, :].data), global_step)
                        writer.add_image('lpg2x2/image/{}'.format(i), normalize_result(1/lpg2x2[i, :, :, :].data), global_step)
                        writer.add_image('lpg4x4/image/{}'.format(i), normalize_result(1/lpg4x4[i, :, :, :].data), global_step)
                        writer.add_image('lpg8x8/image/{}'.format(i), normalize_result(1/lpg8x8[i, :, :, :].data), global_step)
                        writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
                    writer.flush()

                if args.wandb_enable:
                    wandb.log({"loss": loss})
                    wandb.log({'learning_rate': current_lr})
                    wandb.log({'var average': var_sum.item()/var_cnt})

                    output_dim = (int(args.input_width), int(args.input_height))

                    for i in range(num_log_images):
                        wandb.log({
                            "img": [wandb.Image(cv2.resize(inv_normalize(image[i, :, :, :]).data, output_dim), caption='image/image/{}'.format(i))],
                            "depth_truths": [wandb.Image(cv2.resize(normalize_result(1/depth_gt[i, :, :, :].data, output_dim), caption='depth_gt/image/{}'.format(i)))],
                            "depth_preds":  [wandb.Image(cv2.resize(normalize_result(1/depth_est[i, :, :, :].data, output_dim), caption='depth_preds_{}'.format(i)))]})

            if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    checkpoint = {'global_step': global_step,
                                  'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model-{}'.format(global_step))

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()
                eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node)
                if eval_measures is not None:
                    for i in range(9):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                model.train()
                block_print()
                set_misc(model)
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1
       
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()

def main():

    if args.mode != 'train':
        print('bts_main.py is only for training. Use bts_test.py instead.')
        return -1

    model_filename = args.model_name + '.py'
    command = 'mkdir ' + args.log_directory + '/' + args.model_name
    os.system(command)

    if args.checkpoint_path == '':
        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        command = 'cp bts.py ' + model_out_path
        os.system(command)
        aux_out_path = args.log_directory + '/' + args.model_name + '/.'
        command = 'cp bts_main.py ' + aux_out_path
        os.system(command)
        command = 'cp bts_dataloader.py ' + aux_out_path
        os.system(command)
    else:
        loaded_model_dir = os.path.dirname(args.checkpoint_path)
        loaded_model_name = os.path.basename(loaded_model_dir)
        loaded_model_filename = loaded_model_name + '.py'

        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        command = 'cp ' + loaded_model_dir + '/' + loaded_model_filename + ' ' + model_out_path
        os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
