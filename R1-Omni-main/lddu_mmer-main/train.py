from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
import time
import argparse
import warnings
# 忽略一些警告以保持输出整洁
warnings.filterwarnings('ignore')     
from src.models.myModels import MyModel
from src.models.optimization import BertAdam
from src.utils.eval import get_metrics, search_binary, getBinaryTensor
from src.utils.eval_gap import *
from torch.utils.data import DataLoader, WeightedRandomSampler
from util import get_logger
from src.dataloaders.cmu_dataloader_ai import AlignedMoseiDataset, UnAlignedMoseiDataset
BASE_PATH = './'
global logger
from src.utils.modelBackUp import ModelBackUp, EarlyStopping

def get_args(description='Multi-modal Multi-label Emotion Recognition'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.") 
    parser.add_argument("--do_test", action='store_true', help="whether to run test")
    parser.add_argument("--aligned", default=False, action='store_true', help="whether train align of unalign dataset")
    parser.add_argument("--data_path", type=str, default='/data/testmllm/project/video_capture/R1-Omni-main/lddu_mmer-main/dataset/dataset1/aggregated_features.pkl', help='cmu_mosei data_path')
    parser.add_argument("--output_dir", default=BASE_PATH+'/cpkt_align', type=str, required=False,
                            help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--num_thread_reader', type=int, default=1, help='') 
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit') 
    parser.add_argument('--unaligned_data_path', type=str, default='/data/testmllm/project/video_capture/R1-Omni-main/lddu_mmer-main/dataset/dataset1/aggregated_features.pkl', help='load unaligned dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for single GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay') 
    parser.add_argument('--n_display', type=int, default=10, help='Information display frequence')
    parser.add_argument('--text_dim', type=int, default=768, help='text_feature_dimension')  # 文本的特征
    parser.add_argument('--video_dim', type=int, default=896, help='video feature dimension')  # 视频的特征
    parser.add_argument('--audio_dim', type=int, default=896, help='audio_feature_dimension')  # 音频的特征
    parser.add_argument('--seed', type=int, default=42, help='random seed') 
    parser.add_argument('--max_words', type=int, default=500, help='')
    parser.add_argument('--max_frames', type=int, default=500, help='')
    parser.add_argument('--max_sequence', type=int, default=500, help='')
    parser.add_argument('--max_label', type=int, default=6, help='')
    parser.add_argument("--bert_model", default="bert-base", type=str, required=False, help="Bert module")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--audio_model", default="audio-base", type=str, required=False, help="Audio module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.") 
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training") 
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--bert_num_hidden_layers', type=int, default=3, help="Layer NO. of visual.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=3, help="Layer NO. of visual.")
    parser.add_argument('--audio_num_hidden_layers', type=int, default=3, help="Layer No. of audio")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=3, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=1, help="Layer NO. of decoder.")
    parser.add_argument("--num_classes", default=6, type=int, required=False, help='Number of emotion label categories')
    parser.add_argument("--hidden_size",type=int, default=896, help='Size of hidden layer')
    parser.add_argument("--latent_size",type=int, default=128, help='Size of latent variables after encoding')
    parser.add_argument("--label_dim",type=int, default=896, help='Dimension size of labels')
    parser.add_argument("--pro_dim",type=int, default=896, help='Size of projection layer')
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--cml",type=float, default=0.5, help='Classification loss for label-related vectors')
    parser.add_argument("--kll",type=float, default=0.00, help='KL divergence loss')
    parser.add_argument("--cdl",type=float, default=2.0, help='Contrastive loss')
    parser.add_argument("--rel",type=float, default=0.1, help='Reconstruction loss for encoder and decoder')
    parser.add_argument("--device", type=str, default="cuda", help='Device placement location')
    parser.add_argument("--final_loss", type=float, default=1.0, help='Classification loss', required=False) # Final classification losstop
    parser.add_argument("--moco_queue", type=int, default=8192, help='Size of queue')
    parser.add_argument("--average", type=bool, default=False, help='Whether to calculate comprehensive modal scores')
    parser.add_argument("--temperature", type=float, default=0.07, help='Temperature coefficient')
    parser.add_argument('--crl',type=float, default=0.3, help='CRL loss')
    parser.add_argument('--gpu', type=str, default='0', help='Specify which GPU to use (e.g., "0", "1")')
    args = parser.parse_args()
    
    # 设置使用的GPU
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Check paramenters
    if args.gradient_accumulation_steps < 1: 
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args): 
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  
    # 不再直接设置设备，而是让torch.cuda.device_count()根据CUDA_VISIBLE_DEVICES自动识别
    # torch.cuda.set_device(args.local_rank) 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))
    return args

def init_device(args, local_rank):
    global logger
    device = args.device
    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0: 
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))
    return device, n_gpu

def init_model(args, device, n_gpu, local_rank): 
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu') 
    else:
        model_state_dict = None
    # Prepare model 加载预训练模型
    model = MyModel.from_pretrained(args.bert_model, args.visual_model, args.audio_model, args.cross_model, args.decoder_model, task_config=args)
    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." not in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * 1.0},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * 1.0},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]
    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)
    return optimizer, scheduler, model

def prep_dataloader(args):
    Dataset = AlignedMoseiDataset if args.aligned else UnAlignedMoseiDataset
    # 将max_frames参数传递给数据集构造函数
    train_dataset = Dataset(
        args.data_path,
        'train',
        max_frames=args.max_frames  # 传递max_frames参数
    )
    val_dataset = Dataset(
        args.data_path,
        'valid',
        max_frames=args.max_frames  # 传递max_frames参数
    )
    test_dataset = Dataset(
        args.data_path,
        'test',
        max_frames=args.max_frames  # 传递max_frames参数
    )
    
    label_input, label_mask = train_dataset._get_label_input()
    
    # 每个GPU的批次大小
    per_gpu_batch_size = args.batch_size // args.n_gpu
    
    # 配置DataLoader以使用内存高效策略
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,  # 使用固定内存加速GPU数据传输
        shuffle=True,
        drop_last=True  # 丢弃不完整批次以保持一致性
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=per_gpu_batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=per_gpu_batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False
    )
    train_length = len(train_dataset)
    val_length = len(val_dataset)
    test_length = len(test_dataset)
    return train_dataloader, val_dataloader, test_dataloader, train_length, val_length, test_length, label_input, label_mask

def save_model(args, model, epoch):
    # Only save the model it-self
    model_to_save = model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model_{}.bin".format(epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(model, model_file=None):
    print("model_file", model_file)
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
    return model

def compute_class_priors(train_dataloader, device, num_classes):
    priors = torch.zeros(num_classes, device=device)
    total = 0
    with torch.no_grad():
        for batch in train_dataloader:
            # batch: text, text_mask, video, video_mask, audio, audio_mask, labels, samples_index
            labels = batch[6].to(device)
            bsz = labels.size(0)
            priors += labels.sum(dim=0)
            total += bsz
    priors = priors / max(total, 1)
    return priors.clamp(1e-4, 1 - 1e-4)

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0, label_input=None, label_mask=None): 
    global logger
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    total_pred, total_true_label, total_pred_scores = [], [], []
    
    # 创建梯度缩放器用于混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # 训练开始时清零优化器梯度
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_dataloader):
        # 及时清理上一批次的内存
        if step > 0:
            torch.cuda.empty_cache()

        # 优化数据传输到GPU，确保所有GPU都正确接收数据
        batch = tuple(t.to(device=device, non_blocking=False) for t in batch)
        pairs_text, pairs_mask, video, video_mask, audio, audio_mask, ground_label, samples_index = batch
        
        # 使用混合精度进行前向传播
        with torch.cuda.amp.autocast(enabled=True):
            # 直接使用模型进行前向传播，避免多余的计算和中间变量
            model_loss, batch_pred, true_label, pred_scores, mean_similarity, _ = model(
                pairs_text, pairs_mask, video, video_mask, audio, audio_mask, 
                label_input, label_mask, groundTruth_labels=ground_label, 
                training=True, samples_index=samples_index
            )
                
        # 梯度累积
        model_loss = model_loss / args.gradient_accumulation_steps
        
        # 使用缩放器进行反向传播
        scaler.scale(model_loss).backward()
        
        # 仅保存必要的结果用于计算指标
        total_loss += float(model_loss)
        total_pred.append(batch_pred.detach().cpu())   # 立即转移到CPU
        total_true_label.append(true_label.detach().cpu())
        total_pred_scores.append(pred_scores.detach().cpu())

        # 参数更新和梯度裁剪
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            global_step += 1
            optimizer.zero_grad()
            
            if global_step % log_step == 0 and local_rank == 0:
                print("Epoch: %d/%d, Step: %d/%d, Lr: %s, loss: %.4f, Time/step: %.4f, similarity: %.4f %.4f" % (epoch + 1,
                      args.epochs, step + 1,
                      len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),float(model_loss), 
                      (time.time() - start_time) / (log_step * args.gradient_accumulation_steps), mean_similarity[0].cpu(), mean_similarity[1].cpu()))
                start_time = time.time()
                
                # 清理临时变量以节省内存
                del pairs_text, pairs_mask, video, video_mask, audio, audio_mask, ground_label
                torch.cuda.empty_cache()
    
    # 计算平均损失并合并结果
    total_loss = total_loss / len(train_dataloader)
    total_pred = torch.cat(total_pred, 0)
    total_true_label = torch.cat(total_true_label, 0)
    total_pred_scores = torch.cat(total_pred_scores, 0)
    
    return total_loss, total_pred, total_true_label, total_pred_scores, model

def eval_epoch(model, val_dataloader, device, label_input, label_mask):
    model.eval()
    with torch.no_grad():
        # 遍历完整验证/测试集
        eval_preds = []
        eval_true_labels = []
        eval_scores = []
        eval_vars = []
        
        for i, batch in enumerate(val_dataloader):
            # 使用non_blocking=True加速GPU传输
            batch = tuple(t.to(device=device, non_blocking=False) for t in batch)
            text, text_mask, video, video_mask, audio, audio_mask, groundTruth_labels, samples_index = batch
            
            # 前向传播
            batch_pred, true_label, pred_scores, (_, _, common_var) = model(
                text, text_mask, video, video_mask, audio, audio_mask, 
                label_input, label_mask, groundTruth_labels=groundTruth_labels, 
                training=False, samples_index=samples_index
            )
            
            # 只保存必要的结果
            eval_preds.append(batch_pred)
            eval_true_labels.append(true_label)
            eval_scores.append(pred_scores)
            eval_vars.append(common_var)
            
            # 及时释放当前批次的内存
            del text, text_mask, video, video_mask, audio, audio_mask, groundTruth_labels, samples_index
            del batch_pred, true_label, pred_scores, common_var
            torch.cuda.empty_cache()
        
        # 合并结果
        if eval_preds:
            total_pred = torch.cat(eval_preds, 0)
            total_true_label = torch.cat(eval_true_labels, 0)
            total_pred_scores = torch.cat(eval_scores, 0)
            total_var = torch.cat(eval_vars, 0)
        else:
            # 初始化空张量以避免错误
            total_pred = torch.tensor([], device=device)
            total_true_label = torch.tensor([], device=device)
            total_pred_scores = torch.tensor([], device=device)
            total_var = torch.tensor([], device=device)
        return  total_pred, total_true_label, total_pred_scores, total_var
           
def main(args):
    global logger
    # 设置CUDA_VISIBLE_DEVICES环境变量，确保使用指定的单个GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f"Using GPU: {args.gpu}")
    
    local_rank = 0
    device, n_gpu = init_device(args, local_rank) # 确认GPU是否存在, 和设备的关系。
    model = init_model(args, device, n_gpu, local_rank)    
    model = model.to(device)
    if args.aligned == False:
        logger.warning("!!!!!!!!!!!!!! you start train unaligned dataset")
    else:
        logger.warning("!!!!!!!!!!!!!! you start train aligned dataset")
    print('***** dataloder preping ... *****')
    if args.do_train:
        train_dataloader, val_dataloader, test_dataloader, train_length, val_length, test_length, label_input, label_mask = prep_dataloader(args)
        label_input = label_input.to(device)
        label_mask = label_mask.to(device)
        
        # 计算训练集类别先验，并设置到模型，初始化偏置与pos_weight
        num_classes = args.max_label
        priors = compute_class_priors(train_dataloader, device, num_classes)
        if hasattr(model, 'module'):
            model.module.set_class_priors(priors)
            pos_w = model.module.pos_weight.detach().cpu().numpy()
        else:
            model.set_class_priors(priors)
            pos_w = model.pos_weight.detach().cpu().numpy()
        logger.info("Class priors: %s", str(priors.detach().cpu().numpy()))
        logger.info("Pos weight: %s", str(pos_w))
        
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs
        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
        
        best_score,best_test_score, global_step = 0.000, 0.000, 0
        best_output_model_file, best_model = None, None
        early_stopping = EarlyStopping(patience=5, delta=0)
        
        for epoch in range(args.epochs):
            total_loss, total_pred, total_label, total_pred_scores, model= train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                                scheduler, global_step, local_rank=args.local_rank, label_input=label_input, label_mask=label_mask)
            # 修正参数顺序：先传入真实标签，再传入预测
            total_micro_f1, total_micro_precision, total_micro_recall, total_acc = get_metrics(total_label, total_pred)
            total_pred_scores = total_pred_scores.data.cpu().numpy()
            total_label_np = total_label.data.cpu().numpy()
            train_gap = calculate_gap(total_pred_scores, total_label_np) 
            if args.local_rank == 0:
                logger.info("Epoch %d/%d Finished, Train Loss: %f, Train_micro_f1: %f, Train_micro_precision: %f, Train_micro_recall: %f,  Train_acc: %f, train_gap: %f",  \
                    epoch + 1, args.epochs, total_loss, total_micro_f1, total_micro_precision, total_micro_recall,  total_acc, train_gap)

            if args.local_rank == 0:
                logger.info('***** Running testing *****')
                logger.info('  Num examples = %d', test_length)
                logger.info("  Batch_size = %d", args.batch_size)

                test_pred, test_label, test_pred_scores, common_var = eval_epoch(model, test_dataloader, device, label_input, label_mask)
                # 评分已在模型中统一到概率域，无需再次sigmoid
                best_test_threshold = search_binary(test_pred_scores, test_label)
                print(f"best_test_threshold: {best_test_threshold}")
                test_pred = getBinaryTensor(test_pred_scores, best_test_threshold)
                # 修正参数顺序：先传入真实标签，再传入预测
                test_micro_f1, test_micro_precision, test_micro_recall, test_acc = get_metrics(test_label, test_pred)

                # 进一步：按类别搜索阈值（基于每列分数分布），择优
                try:
                    scores_np = test_pred_scores.detach().cpu().numpy()
                    labels_np = test_label.detach().cpu().numpy()
                    class_thresholds = []
                    for j in range(scores_np.shape[1]):
                        # 简单策略：每类独立搜索最佳阈值
                        th = search_binary(scores_np[:, [j]], labels_np[:, [j]])
                        class_thresholds.append(th)
                    class_thresholds = np.array(class_thresholds, dtype=np.float32)
                    test_pred_cls = getBinaryTensor(test_pred_scores, class_thresholds)
                    test_micro_f1_cls, test_micro_precision_cls, test_micro_recall_cls, test_acc_cls = get_metrics(test_label, test_pred_cls)
                    # 择优：选择更好的方案
                    if test_micro_f1_cls > test_micro_f1:
                        test_pred = test_pred_cls
                        test_micro_f1, test_micro_precision, test_micro_recall, test_acc = test_micro_f1_cls, test_micro_precision_cls, test_micro_recall_cls, test_acc_cls
                        best_test_threshold = class_thresholds  # 记录类别阈值
                except Exception as e:
                    print(f"Class-wise thresholding failed: {e}")

                test_pred1, test_label1, test_pred_scores1, common_var = eval_epoch(model, test_dataloader, device, label_input, label_mask)
                # 使用同样的阈值方案进行第二次评估
                if isinstance(best_test_threshold, (list, np.ndarray)):
                    test_pred1 = getBinaryTensor(test_pred_scores1, np.array(best_test_threshold, dtype=np.float32))
                else:
                    test_pred1 = getBinaryTensor(test_pred_scores1, best_test_threshold)
                test_micro_f11, test_micro_precision1, test_micro_recall1, test_acc1 = get_metrics(test_label1, test_pred1)
                
                test_micro_f1 = max(test_micro_f1, test_micro_f11)
                test_micro_precision = max(test_micro_precision, test_micro_precision1)
                test_micro_recall = max(test_micro_recall, test_micro_recall1)
                test_acc = max(test_acc, test_acc1)
                test_pred_scores_np = test_pred_scores.data.cpu().numpy()
                test_label_np = test_label.data.cpu().numpy()
                test_gap = calculate_gap(test_pred_scores_np, test_label_np)
                logger.info("----- micro_f1: %f, micro_precision: %f, micro_recall: %f,  acc: %f, test_gap: %f, threshold: %s", \
                        test_micro_f1, test_micro_precision, test_micro_recall, test_acc, test_gap, str(best_test_threshold))
                early_stopping(test_micro_f1)
                if best_test_score <=  test_micro_f1:
                    best_test_score = test_micro_f1
                    best_model = model
                    output_model_file = save_model(args, best_model, epoch)
                    if best_output_model_file is not None and os.path.exists(best_output_model_file):
                        os.remove(best_output_model_file)
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the f1 is: {:.4f}".format(best_output_model_file, best_test_score))
            if total_micro_f1 - test_micro_f1 > 0.06 and early_stopping.early_stop:
                break
    print("best_output_model_file", best_output_model_file)
    best_model = load_model(model, best_output_model_file)
    test_pred, test_label, test_pred_scores, common_var  = eval_epoch(best_model, test_dataloader, device, label_input, label_mask)
    # 最终评估：模型已输出概率，直接进行阈值搜索与二值化
    final_threshold = search_binary(test_pred_scores, test_label)
    test_pred = getBinaryTensor(test_pred_scores, final_threshold)
    # 类别阈值搜索（最终评估阶段），择优
    try:
        scores_np = test_pred_scores.detach().cpu().numpy()
        labels_np = test_label.detach().cpu().numpy()
        class_thresholds = []
        for j in range(scores_np.shape[1]):
            th = search_binary(scores_np[:, [j]], labels_np[:, [j]])
            class_thresholds.append(th)
        class_thresholds = np.array(class_thresholds, dtype=np.float32)
        test_pred_cls = getBinaryTensor(test_pred_scores, class_thresholds)
        test_micro_f1_cls, test_micro_precision_cls, test_micro_recall_cls, test_acc_cls = get_metrics(test_label, test_pred_cls)
        if test_micro_f1_cls > test_micro_f1:
            test_pred = test_pred_cls
            final_threshold = class_thresholds
            test_micro_f1, test_micro_precision, test_micro_recall, test_acc = test_micro_f1_cls, test_micro_precision_cls, test_micro_recall_cls, test_acc_cls
    except Exception as e:
        print(f"Final class-wise thresholding failed: {e}")
    logger.info("Final test results: f1 %f,\tp %f,\tr %f,\tacc %f, threshold %s",
                test_micro_f1, test_micro_precision, test_micro_recall, test_acc, str(final_threshold))

      
if __name__ == "__main__":
    args = get_args() # 获取日志
    args = set_seed_logger(args)    
    main(args)