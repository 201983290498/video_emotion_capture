from sklearn import metrics
import numpy as np
import torch
from sklearn import metrics
import logging
logger = logging.getLogger(__name__)

def dense(y):
    label_y = []
    for i in range(len(y)):
        for j in range(len(y[i])):
            label_y.append(y[i][j])

    return label_y

def get_accuracy(y, y_pre):
    """
    计算准确率，支持各种输入类型和形状
    
    Args:
        y: 真实标签（numpy数组、torch张量或标量）
        y_pre: 预测结果（numpy数组、torch张量或标量）
    
    Returns:
        准确率（0-1之间的浮点数）
    """
    try:
        # 确保输入是numpy数组或tensor
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        if isinstance(y_pre, torch.Tensor):
            y_pre = y_pre.cpu().detach().numpy()
            
        # 检查是否是标量并转换为2D数组
        if np.isscalar(y):
            y = np.array([[y]])
        elif isinstance(y, np.ndarray):
            # 确保y_pre是数组类型
            if not isinstance(y_pre, np.ndarray):
                y_pre = np.array([[y_pre]] * len(y))
        
        if np.isscalar(y_pre):
            y_pre = np.array([[y_pre]] * len(y) if isinstance(y, np.ndarray) and hasattr(y, '__len__') else [[y_pre]])
        
        # 确保数据类型是float32
        if isinstance(y, np.ndarray) and y.dtype == np.float16:
            y = y.astype(np.float32)
        if isinstance(y_pre, np.ndarray) and y_pre.dtype == np.float16:
            y_pre = y_pre.astype(np.float32)
        
        # 检查数据形状，处理不同维度的输入
        try:
            if isinstance(y, np.ndarray) and len(y.shape) == 1:
                # 一维数据，添加维度以保持一致性
                y = y.reshape(-1, 1)
            if isinstance(y_pre, np.ndarray) and len(y_pre.shape) == 1:
                y_pre = y_pre.reshape(-1, 1)
        except Exception as e:
            logger.warning(f"调整数据形状时出错: {e}")
        
        # 确保y和y_pre形状一致
        try:
            if isinstance(y, np.ndarray) and isinstance(y_pre, np.ndarray) and y.shape != y_pre.shape:
                # 尝试广播到相同形状
                # 首先尝试扩展到相同的维度
                if len(y.shape) > len(y_pre.shape):
                    # 扩展y_pre的维度
                    y_pre = np.expand_dims(y_pre, axis=tuple(range(len(y.shape) - len(y_pre.shape))))
                elif len(y_pre.shape) > len(y.shape):
                    # 扩展y的维度
                    y = np.expand_dims(y, axis=tuple(range(len(y_pre.shape) - len(y.shape))))
                
                # 尝试广播到相同形状
                try:
                    # 创建兼容形状的新数组
                    max_shape = np.maximum(y.shape, y_pre.shape)
                    y_broadcast = np.zeros(max_shape, dtype=y.dtype)
                    y_pre_broadcast = np.zeros(max_shape, dtype=y_pre.dtype)
                    
                    # 计算切片
                    y_slices = tuple(slice(0, s) for s in y.shape)
                    y_pre_slices = tuple(slice(0, s) for s in y_pre.shape)
                    
                    # 填充数据
                    y_broadcast[y_slices] = y
                    y_pre_broadcast[y_pre_slices] = y_pre
                    
                    y, y_pre = y_broadcast, y_pre_broadcast
                except Exception as e:
                    logger.warning(f"无法广播y和y_pre到相同形状: {y.shape} vs {y_pre.shape}, 错误: {e}")
                    # 如果广播失败，尝试其他方法
                    if hasattr(y, '__len__') and hasattr(y_pre, '__len__') and len(y) > 0 and len(y_pre) > 0:
                        # 取最小公倍数长度
                        min_len = min(len(y), len(y_pre))
                        y = y[:min_len]
                        y_pre = y_pre[:min_len]
                    else:
                        return 0.0
        except Exception as e:
            logger.warning(f"对齐y和y_pre形状时出错: {e}")
        
        # 计算准确率
        try:
            # 确保y和y_pre都是数组
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if not isinstance(y_pre, np.ndarray):
                y_pre = np.array(y_pre)
            
            # 计算样本数
            if hasattr(y, '__len__'):
                sambles = len(y)
            else:
                sambles = 1
            
            count = 0.0
            
            # 处理不同维度的输入
            if hasattr(y, 'shape') and hasattr(y_pre, 'shape'):
                if (len(y.shape) == 1 or y.shape[1] == 1) and (len(y_pre.shape) == 1 or y_pre.shape[1] == 1):
                    # 一维或单列数据
                    for i in range(sambles):
                        try:
                            # 获取当前样本的值
                            y_val = y[i] if hasattr(y, '__len__') and i < len(y) else y
                            y_pre_val = y_pre[i] if hasattr(y_pre, '__len__') and i < len(y_pre) else y_pre
                            
                            # 确保值是标量
                            if hasattr(y_val, '__len__') and len(y_val) > 0:
                                y_val = y_val[0]
                            if hasattr(y_pre_val, '__len__') and len(y_pre_val) > 0:
                                y_pre_val = y_pre_val[0]
                            
                            # 计算真阳性和总相关样本
                            y_true = 1 if (float(y_val) > 0 and float(y_pre_val) > 0) else 0
                            all_y = 1 if (float(y_val) > 0 or float(y_pre_val) > 0) else 0
                            
                            if all_y <= 0:
                                all_y = 1
                            
                            count += float(y_true) / float(all_y)
                        except Exception as e:
                            logger.warning(f"处理样本{i}时出错: {e}")
                            continue
                else:
                    # 多维数据（多类别）
                    for i in range(sambles):
                        try:
                            y_true = 0
                            all_y = 0
                            
                            # 确保i在有效范围内
                            if hasattr(y, '__len__') and i >= len(y):
                                continue
                            if hasattr(y_pre, '__len__') and i >= len(y_pre):
                                continue
                            
                            # 获取当前样本
                            y_sample = y[i] if hasattr(y, '__len__') else y
                            y_pre_sample = y_pre[i] if hasattr(y_pre, '__len__') else y_pre
                            
                            # 遍历所有类别
                            num_classes = 0
                            if hasattr(y_sample, '__len__') and hasattr(y_pre_sample, '__len__'):
                                num_classes = min(len(y_sample), len(y_pre_sample))
                            
                            for j in range(num_classes):
                                try:
                                    # 获取类别值
                                    y_cls = y_sample[j] if hasattr(y_sample, '__len__') and j < len(y_sample) else y_sample
                                    y_pre_cls = y_pre_sample[j] if hasattr(y_pre_sample, '__len__') and j < len(y_pre_sample) else y_pre_sample
                                    
                                    # 确保值是标量
                                    if hasattr(y_cls, '__len__') and len(y_cls) > 0:
                                        y_cls = y_cls[0]
                                    if hasattr(y_pre_cls, '__len__') and len(y_pre_cls) > 0:
                                        y_pre_cls = y_pre_cls[0]
                                    
                                    # 计算真阳性和总相关样本
                                    if float(y_cls) > 0 and float(y_pre_cls) > 0:
                                        y_true += 1
                                    if float(y_cls) > 0 or float(y_pre_cls) > 0:
                                        all_y += 1
                                except Exception as e:
                                    logger.warning(f"处理样本{i}类别{j}时出错: {e}")
                                    continue
                            
                            if all_y <= 0:
                                all_y = 1
                            
                            count += float(y_true) / float(all_y)
                        except Exception as e:
                            logger.warning(f"处理样本{i}时出错: {e}")
                            continue
            
            # 计算最终准确率
            if sambles > 0:
                acc = float(count) / float(sambles)
                acc = round(acc, 4)
                return acc
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"计算准确率时出错: {e}")
            # 尝试使用简单方法计算准确率
            try:
                # 将输入展平
                y_flat = np.ravel(y) if isinstance(y, np.ndarray) else np.array([y])
                y_pre_flat = np.ravel(y_pre) if isinstance(y_pre, np.ndarray) else np.array([y_pre])
                
                # 确保长度一致
                min_len = min(len(y_flat), len(y_pre_flat))
                y_flat = y_flat[:min_len]
                y_pre_flat = y_pre_flat[:min_len]
                
                # 计算准确率
                correct = np.sum((y_flat > 0) & (y_pre_flat > 0))
                total = np.sum((y_flat > 0) | (y_pre_flat > 0))
                
                if total > 0:
                    return round(float(correct) / float(total), 4)
                else:
                    return 0.0
            except Exception as e2:
                logger.error(f"备用准确率计算方法也失败: {e2}")
                return 0.0
    except Exception as e:
        logger.error(f"Error calculating accuracy: {str(e)}, y type: {type(y)}, y_pre type: {type(y_pre)}")
        return 0.0

def get_metrics(y, y_pre):
    """
    计算评估指标，处理不同数据类型和形状的输入
    
    Args:
        y: 真实标签（numpy数组、torch张量或标量）
        y_pre: 预测结果（numpy数组、torch张量或标量）
        
    Returns:
        micro_f1, micro_precision, micro_recall, acc
    """
    try:
        # 确保输入是numpy数组
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        if isinstance(y_pre, torch.Tensor):
            y_pre = y_pre.cpu().detach().numpy()
        
        # 检查是否是标量并转换为数组
        if np.isscalar(y):
            y = np.array([y])
        if np.isscalar(y_pre):
            y_pre = np.array([y_pre])
        
        # 确保y和y_pre是数组类型
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y)
            except Exception as e:
                logger.warning(f"无法将y转换为numpy数组: {e}")
                y = np.array([0])  # 提供默认值以避免后续错误
        
        if not isinstance(y_pre, np.ndarray):
            try:
                y_pre = np.array(y_pre)
            except Exception as e:
                logger.warning(f"无法将y_pre转换为numpy数组: {e}")
                y_pre = np.array([0])  # 提供默认值以避免后续错误
        
        # 添加调试日志，查看预测值分布
        if isinstance(y_pre, np.ndarray):
            try:
                logger.debug("y_pre statistics - Min: %.4f, Max: %.4f, Mean: %.4f, Median: %.4f",
                           np.min(y_pre), np.max(y_pre), np.mean(y_pre), np.median(y_pre))
                logger.debug("y_pre positive rate: %.4f", np.mean(y_pre > 0))
            except Exception as e:
                logger.warning(f"计算y_pre统计信息时出错: {e}")
        if isinstance(y, np.ndarray):
            try:
                logger.debug("y_true positive rate: %.4f", np.mean(y > 0))
            except Exception as e:
                logger.warning(f"计算y统计信息时出错: {e}")
        
        # 确保数据类型是scipy.sparse支持的类型
        try:
            if isinstance(y, np.ndarray) and y.dtype == np.float16:
                y = y.astype(np.float32)
            if isinstance(y_pre, np.ndarray) and y_pre.dtype == np.float16:
                y_pre = y_pre.astype(np.float32)
        except Exception as e:
            logger.warning(f"数据类型转换时出错: {e}")
        
        # 确保形状兼容
        try:
            if isinstance(y, np.ndarray) and isinstance(y_pre, np.ndarray) and y.shape != y_pre.shape:
                # 尝试广播或调整形状
                if len(y.shape) > len(y_pre.shape):
                    # 扩展y_pre的维度
                    y_pre = np.expand_dims(y_pre, axis=tuple(range(len(y.shape) - len(y_pre.shape))))
                elif len(y_pre.shape) > len(y.shape):
                    # 扩展y的维度
                    y = np.expand_dims(y, axis=tuple(range(len(y_pre.shape) - len(y.shape))))
                
                # 尝试调整长度
                if hasattr(y, '__len__') and hasattr(y_pre, '__len__') and len(y) != len(y_pre):
                    min_len = min(len(y), len(y_pre))
                    y = y[:min_len]
                    y_pre = y_pre[:min_len]
        except Exception as e:
            logger.warning(f"调整y和y_pre形状时出错: {e}")
        
        # 计算准确率
        acc = get_accuracy(y, y_pre)
        
        # 初始化指标值
        micro_f1 = 0.0
        micro_precision = 0.0
        micro_recall = 0.0
        
        # 计算其他指标，确保y和y_pre是有效的数组
        try:
            # 确保输入是有效的二维数组
            if isinstance(y, np.ndarray) and isinstance(y_pre, np.ndarray):
                # 检查数据维度，确保适合metrics函数
                if len(y.shape) == 1:
                    y = y.reshape(-1, 1)
                if len(y_pre.shape) == 1:
                    y_pre = y_pre.reshape(-1, 1)
                
                # 检查是否是空数组
                if len(y) == 0 or len(y_pre) == 0:
                    logger.warning("Empty arrays provided to metrics functions")
                    return micro_f1, micro_precision, micro_recall, acc
                
                # 尝试计算micro_f1
                try:
                    micro_f1 = metrics.f1_score(y, y_pre, average='micro')
                except Exception as e:
                    logger.error(f"Error calculating micro_f1: {e}")
                    micro_f1 = 0.0
                
                # 尝试计算micro_precision
                try:
                    micro_precision = metrics.precision_score(y, y_pre, average='micro')
                except Exception as e:
                    logger.error(f"Error calculating micro_precision: {e}")
                    micro_precision = 0.0
                
                # 尝试计算micro_recall
                try:
                    micro_recall = metrics.recall_score(y, y_pre, average='micro')
                except Exception as e:
                    logger.error(f"Error calculating micro_recall: {e}")
                    micro_recall = 0.0
        except Exception as e:
            logger.error(f"Error in metrics calculation: {e}")
        
        return micro_f1, micro_precision, micro_recall, acc
    except Exception as e:
        logger.error(f"Error in get_metrics: {str(e)}, y type: {type(y)}, y_pre type: {type(y_pre)}")
        return 0.0, 0.0, 0.0, 0.0


def getBinaryTensor(imgTensor, boundary = 0.21, adaptive=False):
    """
    将预测分数转换为二值标签
    
    Args:
        imgTensor: 预测分数张量或数组
        boundary: 阈值（可以是单个值或类别特定阈值列表）
        adaptive: 是否使用自适应阈值
    
    Returns:
        二值化后的张量
    """
    try:
        # 确保输入是tensor
        if isinstance(imgTensor, np.ndarray):
            # 检查数据类型并转换为float32（避免scipy.sparse不支持float16的问题）
            if imgTensor.dtype == np.float16:
                imgTensor = imgTensor.astype(np.float32)
            imgTensor = torch.tensor(imgTensor)
        elif isinstance(imgTensor, (int, float, np.float32, np.float64)):
            # 处理标量输入
            imgTensor = torch.tensor([[imgTensor]], dtype=torch.float32)
        elif not isinstance(imgTensor, torch.Tensor):
            # 尝试转换为tensor
            try:
                imgTensor = torch.tensor(imgTensor, dtype=torch.float32)
            except Exception as e:
                logger.warning(f"无法将输入转换为tensor: {e}")
                return torch.zeros_like(torch.tensor([[0.0]]), dtype=torch.float32)
        
        # 确保数据类型是float32
        if imgTensor.dtype != torch.float32:
            imgTensor = imgTensor.type(torch.float32)
        
        # 检查是否是类别特定阈值（列表或数组）
        is_class_specific = isinstance(boundary, (list, np.ndarray, torch.Tensor))
        
        # 如果是自适应模式，根据预测分数的分布动态计算阈值
        if adaptive:
            # 将预测分数展平
            flat_preds = imgTensor.flatten()
            
            # 计算预测分数的分布特征
            try:
                pred_min = torch.min(flat_preds)
                pred_max = torch.max(flat_preds)
                pred_mean = torch.mean(flat_preds)
                pred_std = torch.std(flat_preds)
            except Exception as e:
                logger.warning(f"Error calculating prediction statistics: {e}")
                pred_min, pred_max, pred_mean, pred_std = 0.0, 1.0, 0.5, 0.2
            
            # 根据预测分数的分布动态调整阈值
            try:
                if (pred_max - pred_min) < 0.5:
                    # 对于范围较小的预测分数，使用均值+0.8*标准差作为阈值
                    boundary = max(pred_min, min(pred_mean + 0.8 * pred_std, pred_max))
                else:
                    # 对于范围较大的预测分数，使用0.4分位数作为阈值
                    k = int(0.4 * len(flat_preds))
                    sorted_preds = torch.sort(flat_preds)[0]
                    boundary = sorted_preds[k] if k < len(sorted_preds) else sorted_preds[-1]
                
                # 确保阈值不低于一个最小合理值
                boundary = max(boundary, 0.1)
                logger.info(f"使用自适应阈值: {boundary:.4f} (预测分数分布: Min={pred_min:.4f}, Max={pred_max:.4f}, Mean={pred_mean:.4f})")
            except Exception as e:
                logger.warning(f"Error calculating adaptive threshold: {e}")
                boundary = 0.5
                
            # 自适应模式下不使用类别特定阈值
            is_class_specific = False
        
        # 创建二元张量
        one = torch.ones_like(imgTensor, dtype=torch.float32).fill_(1)
        zero = torch.zeros_like(imgTensor, dtype=torch.float32).fill_(0)
        
        # 使用torch.where进行二值化
        if is_class_specific:
            # 类别特定阈值模式
            logger.debug(f"应用类别特定阈值，输入形状: {imgTensor.shape}")
            # 确保boundary是tensor
            if isinstance(boundary, (list, np.ndarray)):
                # 确保阈值类型是float32
                if isinstance(boundary, np.ndarray) and boundary.dtype == np.float16:
                    boundary = boundary.astype(np.float32)
                boundary = torch.tensor(boundary, dtype=torch.float32)
                
            # 确保boundary在正确的设备上
            if boundary.device != imgTensor.device:
                boundary = boundary.to(imgTensor.device)
                
            # 确保imgTensor至少是2D（batch_size, num_classes）
            if len(imgTensor.shape) == 1:
                imgTensor = imgTensor.unsqueeze(0)
                one = one.unsqueeze(0)
                zero = zero.unsqueeze(0)
            
            # 对于每个类别应用对应的阈值
            binary_tensor = torch.zeros_like(imgTensor, dtype=torch.float32)
            try:
                # 确保阈值数量足够
                num_classes = imgTensor.shape[1]
                if len(boundary) < num_classes:
                    # 如果阈值数量不足，复制最后一个阈值
                    extended_boundary = torch.zeros(num_classes, device=boundary.device)
                    extended_boundary[:len(boundary)] = boundary
                    extended_boundary[len(boundary):] = boundary[-1]
                    boundary = extended_boundary
                    logger.warning(f"阈值数量不足({len(boundary)} < {num_classes})，使用最后一个阈值填充")
                elif len(boundary) > num_classes:
                    # 如果阈值数量过多，截断
                    boundary = boundary[:num_classes]
                    logger.warning(f"阈值数量过多({len(boundary)} > {num_classes})，截断到类别数量")
                
                # 优化的批量处理方式，避免循环
                binary_tensor = torch.where(imgTensor > boundary.unsqueeze(0), one, zero)
                logger.debug(f"类别特定阈值应用成功，输出形状: {binary_tensor.shape}")
            except Exception as e:
                logger.warning(f"Error applying class-specific thresholds: {e}")
                # 出错时默认使用0.5作为全局阈值
                binary_tensor = torch.where(imgTensor > 0.5, one, zero)
        else:
            # 单阈值模式
            try:
                # 确保阈值是float32
                if isinstance(boundary, torch.Tensor) and boundary.dtype != torch.float32:
                    boundary = boundary.type(torch.float32)
                elif isinstance(boundary, (int, float)):
                    boundary = float(boundary)
                
                binary_tensor = torch.where(imgTensor > boundary, one, zero)
            except Exception as e:
                logger.warning(f"Error applying single threshold: {e}")
                # 出错时默认使用0.5作为阈值
                binary_tensor = torch.where(imgTensor > 0.5, one, zero)
        
        # 返回二值化结果
        return binary_tensor
    except Exception as e:
        logger.error(f"Error in getBinaryTensor: {e}")
        # 发生错误时返回默认的零张量
        if isinstance(imgTensor, torch.Tensor):
            return torch.zeros_like(imgTensor, dtype=torch.float32)
        elif isinstance(imgTensor, np.ndarray):
            return np.zeros_like(imgTensor, dtype=np.float32)
        else:
            return torch.tensor([[0.0]], dtype=torch.float32)


def search_binary(pred_score, total_label, eval_threshold=None):
    """
    搜索最佳二值化阈值（全局单阈值）。
    
    参数:
        pred_score: 预测分数张量或数组
        total_label: 真实标签张量或数组
        eval_threshold: 可选的评估阈值（仅日志评估，不调整搜索）
    
    返回:
        最佳单一阈值（float）
    """
    try:
        # 统一为 numpy 数组
        if isinstance(pred_score, torch.Tensor):
            pred_score_np = pred_score.cpu().detach().numpy()
        elif isinstance(pred_score, np.ndarray):
            pred_score_np = pred_score
        else:
            pred_score_np = np.array(pred_score)
        
        if isinstance(total_label, torch.Tensor):
            total_label_np = total_label.cpu().detach().numpy()
        elif isinstance(total_label, np.ndarray):
            total_label_np = total_label
        else:
            total_label_np = np.array(total_label)
        
        # 转换 float16 -> float32，保证兼容性
        if hasattr(pred_score_np, 'dtype') and pred_score_np.dtype == np.float16:
            pred_score_np = pred_score_np.astype(np.float32)
        if hasattr(total_label_np, 'dtype') and total_label_np.dtype == np.float16:
            total_label_np = total_label_np.astype(np.float32)
        
        # 保证二维形状 (batch, classes)
        if pred_score_np.ndim == 1:
            pred_score_np = pred_score_np.reshape(-1, 1)
        if total_label_np.ndim == 1:
            total_label_np = total_label_np.reshape(-1, 1)
        
        # 评估辅助函数，统一调用顺序为 (y_true, y_pred)
        def eval_at(threshold: float):
            total_pred = getBinaryTensor(pred_score_np, threshold)
            if isinstance(total_pred, torch.Tensor):
                total_pred = total_pred.cpu().numpy()
            if isinstance(total_pred, np.ndarray) and total_pred.dtype == np.float16:
                total_pred = total_pred.astype(np.float32)
            return get_metrics(total_label_np, total_pred)
        
        # 若提供了评估阈值，打印一次评估日志
        if eval_threshold is not None:
            try:
                f1_e, p_e, r_e, acc_e = eval_at(float(eval_threshold))
                logger.info(
                    "Eval Threshold, Train_micro_f1: %f, Train_micro_precision: %f, Train_micro_recall: %f, Train_acc: %f",
                    f1_e, p_e, r_e, acc_e
                )
            except Exception as e:
                logger.warning(f"Error evaluating provided threshold: {e}")
        
        # 若分数不在[0,1]，视为logits并转换到概率域
        try:
            raw_min = float(np.min(pred_score_np))
            raw_max = float(np.max(pred_score_np))
        except Exception:
            raw_min, raw_max = 0.0, 1.0
        if raw_min < 0.0 or raw_max > 1.0:
            try:
                pred_score_np = 1.0 / (1.0 + np.exp(-pred_score_np))
                logger.info("已将logits转换为概率域(使用sigmoid)")
            except Exception as e:
                logger.warning(f"Logits->概率转换失败: {e}")
        
        # 概率域分布统计
        try:
            min_score = float(np.min(pred_score_np))
            max_score = float(np.max(pred_score_np))
            mean_score = float(np.mean(pred_score_np))
            median_score = float(np.median(pred_score_np))
            std_score = float(np.std(pred_score_np))
            logger.info(
                f"预测分数分布: Min={min_score:.4f}, Max={max_score:.4f}, Mean={mean_score:.4f}, Median={median_score:.4f}, Std={std_score:.4f}"
            )
        except Exception as e:
            logger.warning(f"Error calculating score statistics: {e}")
            min_score, max_score, mean_score, median_score, std_score = 0.0, 1.0, 0.5, 0.5, 0.25
        
        # 概率域搜索范围
        start_threshold = max(0.01, min_score - 0.1)
        end_threshold = min(0.99, max_score + 0.1)
        
        # 候选阈值集合
        thresholds = []
        thresholds.extend([mean_score, median_score])
        # 关键区间精细搜索 0.1-0.3
        if start_threshold < 0.3 and end_threshold > 0.1:
            seg_start = max(start_threshold, 0.1)
            seg_end = min(0.3, end_threshold)
            num_points = max(2, int((seg_end - seg_start) / 0.001) + 1)
            try:
                thresholds.extend(np.linspace(seg_start, seg_end, num=num_points).tolist())
            except Exception as e:
                logger.warning(f"Error creating segment thresholds: {e}")
        # 全局均匀点
        try:
            thresholds.extend(np.linspace(start_threshold, end_threshold, num=20).tolist())
        except Exception as e:
            logger.warning(f"Error creating global thresholds: {e}")
        # 分位数
        try:
            thresholds.extend(np.percentile(pred_score_np, [10, 25, 50, 75, 90]).tolist())
        except Exception:
            pass
        # 标准差相关
        thresholds.extend([
            mean_score - 0.5 * std_score, mean_score + 0.5 * std_score,
            mean_score - std_score, mean_score + std_score
        ])
        
        # 去重、裁剪与采样
        try:
            thresholds = np.array(thresholds, dtype=np.float32)
            thresholds = thresholds[(thresholds >= 0.0) & (thresholds <= 1.0)]
            thresholds = np.unique(thresholds)
            if len(thresholds) == 0:
                thresholds = np.array([0.5], dtype=np.float32)
            elif len(thresholds) > 100:
                idx = np.linspace(0, len(thresholds) - 1, num=100, dtype=int)
                thresholds = thresholds[idx]
        except Exception as e:
            logger.warning(f"Error processing thresholds: {e}")
            thresholds = np.array([0.5], dtype=np.float32)
        
        # 搜索最佳阈值
        best_f1 = -1.0
        best_threshold = float(thresholds[0])
        best_result = (0.0, 0.0, 0.0, 0.0)
        for th in thresholds:
            try:
                f1, p, r, acc = eval_at(float(th))
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(th)
                    best_result = (f1, p, r, acc)
            except Exception as e:
                logger.warning(f"Error evaluating threshold {th}: {e}")
        
        # 局部精细化搜索
        try:
            fine_thresholds = np.linspace(max(0.0, best_threshold - 0.02), min(1.0, best_threshold + 0.02), num=30)
            for th in fine_thresholds:
                f1, p, r, acc = eval_at(float(th))
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(th)
                    best_result = (f1, p, r, acc)
        except Exception as e:
            logger.warning(f"Error in local fine search: {e}")
        
        logger.info(
            f"最佳阈值: {best_threshold:.4f}, Train_micro_f1: {best_result[0]:.4f}, Train_micro_precision: {best_result[1]:.4f}, Train_micro_recall: {best_result[2]:.4f}, Train_acc: {best_result[3]:.4f}"
        )
        
        # 简要建议
        if best_result[0] < 0.5:
            if (max_score - min_score) < 0.3:
                logger.info("警告: 预测分数分布过于集中，考虑调整正则化或学习率。")
            elif best_threshold < 0.2:
                logger.info("提示: 最佳阈值较低，考虑提升置信度或增加正类权重。")
            elif best_threshold > 0.8:
                logger.info("提示: 最佳阈值较高，考虑提高召回率或数据增强。")
        
        return best_threshold
    except Exception as e:
        logger.error(f"Error in search_binary: {e}")
        return 0.5