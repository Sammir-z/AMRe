# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class AME(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.kl = nn.KLDivLoss(reduction="none", log_target=False)
#         # self.ce_loss = nn.BCEWithLogitsLoss(reduction="none")  # reduction=none 表示的是忽略掉batch_size维度的
#         self.ce_loss = nn.CrossEntropyLoss(reduction="none")
#         self.beta = args.ame_beta
#         self.gama = args.ame_gama
#         self.temperature = args.ame_temperature
#         self.un_target_D = 1/args.num_classes
#         self.model_name = eval(args.model_name)

#     def forward(self, p, q, labels=None, epoch_index=-1,epoch=-1):
#         """
#         一定要确保传进来的是logit
#         """
#         # print(f"p:{p}")
#         # print(f"q:{q}")
#         # print(f"self.model_name is {self.model_name} ")
#         # print(f"self.model_name is {eval(self.model_name)} ")
#         batch_size = p.shape[0]
#         eps = 1e-8
#         # ================先计算出准确度的分数====================
#         labels = labels.long()
#         p_score = self.ce_loss(p, labels).squeeze()
#         q_score = self.ce_loss(q, labels).squeeze()

#         # 使用softmax归一化准确度分数（BCE越小越好，取负号）
#         acc_scores = torch.stack([p_score, q_score], dim=-1)  # [batch, 2]
#         acc_softmax = torch.softmax(acc_scores/self.temperature, dim=-1)  # [batch, 2]
#         p_ratio = acc_softmax[:, 0]
#         q_ratio = acc_softmax[:, 1]
#         # print(f"p_ratio.shape is {p_ratio.shape}")
        
#         # =============计算出不确定性的分数===============
#         log_softmax_p = torch.log_softmax(p, dim=-1)
#         log_softmax_q = torch.log_softmax(q, dim=-1)
#         target_dis = torch.full_like(log_softmax_p, torch.tensor(self.un_target_D))
#         P_K_softmax = self.kl(log_softmax_p, target_dis).sum(dim=-1)
#         Q_K_softmax = self.kl(log_softmax_q, target_dis).sum(dim=-1)
#         uncerten_socres = torch.stack([-P_K_softmax, -Q_K_softmax], dim=-1)  # [batch, 2]
#         uncerten_softmax = torch.softmax(uncerten_socres/self.temperature, dim=-1)
#         p_uncor_ratio = uncerten_softmax[:,0]
#         q_uncor_ratio = uncerten_softmax[:,1]

#         # =================做结合=================
#         Image_ratio = p_ratio * self.beta + p_uncor_ratio * (1 - self.beta)
#         Text_ratio = q_ratio * self.beta + q_uncor_ratio * (1 - self.beta)

#         IT_ratio = torch.cat([Image_ratio.unsqueeze(1), Text_ratio.unsqueeze(1)], dim=-1)
#         # 利用softmax归一化
#         # IT_ratio = torch.softmax(IT_ratio,dim=-1)
#         # 计算IT_ratio中两列的差异绝对值的平均值
#         # mean_absolute_difference = torch.mean(torch.abs(IT_ratio[:, 0] - IT_ratio[:, 1]))
#         # print(f"mean_absolute_difference is {mean_absolute_difference}")
#         # gama = min(self.gama, mean_absolute_difference.item())
#         gama = self.gama
#         result = torch.ones_like(IT_ratio)
#         for i in range(batch_size):
#             if abs(IT_ratio[i, 0] - IT_ratio[i, 1]) >= gama:
#                 # print(f"{i} IT argmin is {torch.argmin(IT_ratio[i])}")
#                 result[i, torch.argmin(IT_ratio[i])] = 0
        
#         # =======================做输出====================
#         if epoch_index == 0:
#             # 确保所有张量都是1维的
#             tensors = {
#                 "labels": labels.squeeze(),
#                 "p_pre": torch.softmax(p,dim=-1),
#                 "q_pre": torch.softmax(q,dim=-1),
#                 "p_score": p_score.squeeze(),
#                 "q_score": q_score.squeeze(),
#                 "p_acc_ratio": p_ratio.squeeze(),
#                 "q_acc_ratio": q_ratio.squeeze(),
#                 "p_unc_ratio": p_uncor_ratio.squeeze(),
#                 "q_unc_ratio": q_uncor_ratio.squeeze(),
#                 f"{self.model_name[0]}_final_ratio": IT_ratio[:,0].squeeze(),
#                 f"{self.model_name[1]}_final_ratio": IT_ratio[:,1].squeeze()
#             }

#             # 打印表头 - 优化版本
#             print("\n" + "=" * 180)
#             print(f"{'Epoch 0 Detailed Statistics':^180}")
#             print("=" * 180)
#             header = (
#                 f"{'Sample':<6} | {'Label':<6} | "
#                 f"{self.model_name[0]+'_Prob':<12} | {self.model_name[1]+'_Prob':<12} | "
#                 f"{self.model_name[0]+'_Loss':<12} | {self.model_name[1]+'_Loss':<12} | "
#                 f"{self.model_name[0]+'_AccR':<12} | {self.model_name[1]+'_AccR':<12} | "
#                 f"{self.model_name[0]+'_UncR':<12} | {self.model_name[1]+'_UncR':<12} | "
#                 f"{self.model_name[0]+'_FinalR':<14} | {self.model_name[1]+'_FinalR':<14} | "
#                 f"{'Decision':<12}"
#             )
#             print(header)
#             print("-" * 180)

#             # 打印每个样本的数据
#             for i in range(len(tensors["p_pre"])):
#                 label_val = int(tensors['labels'][i].item())
#                 row = f"{i:<6} | {label_val:<6} | "
#                 row += f"{tensors['p_pre'][i][label_val].item():<12.4f} | "
#                 row += f"{tensors['q_pre'][i][label_val].item():<12.4f} | "
#                 row += f"{tensors['p_score'][i].item():<12.4f} | "
#                 row += f"{tensors['q_score'][i].item():<12.4f} | "
#                 row += f"{tensors['p_acc_ratio'][i].item():<12.4f} | "
#                 row += f"{tensors['q_acc_ratio'][i].item():<12.4f} | "
#                 row += f"{tensors['p_unc_ratio'][i].item():<12.4f} | "
#                 row += f"{tensors['q_unc_ratio'][i].item():<12.4f} | "
#                 row += f"{tensors[f'{self.model_name[0]}_final_ratio'][i].item():<14.4f} | "
#                 row += f"{tensors[f'{self.model_name[1]}_final_ratio'][i].item():<14.4f} | "

#                 # 添加决策列
#                 decision = "None"
#                 if result[i, 1] == 0:
#                     decision = self.model_name[1]
#                 elif result[i, 0] == 0:
#                     decision = self.model_name[0]
#                 row += f"{decision:<12}"

#                 print(row)

#             print("\n" + "=" * 180)

#         return result.bool(), IT_ratio
    

# =============================任意数量模态融合================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class AME(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="none", log_target=False)
        # self.ce_loss = nn.BCEWithLogitsLoss(reduction="none")  # reduction=none 表示的是忽略掉batch_size维度的
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.beta = args.ame_beta
        self.gama = args.ame_gama
        self.temperature = args.ame_temperature
        self.un_target_D = 1/args.num_classes
        self.model_name = eval(args.model_name)

    def forward(self, *modalities, labels=None, epoch_index=-1, epoch=-1):
        """
        接受任意数量的模态作为输入，每个模态都是一个logit张量。
        """
        if not modalities:
            raise ValueError("At least one modality must be provided.")

        num_modalities = len(modalities)
        batch_size = modalities[0].shape[0]
        device = modalities[0].device
        eps = 1e-8
        
        # ================ 1. 计算准确度分数 ====================
        labels = labels.long()
        # 计算每个模态的交叉熵损失
        acc_scores_list = [self.ce_loss(m, labels).squeeze() for m in modalities]
        acc_scores = torch.stack(acc_scores_list, dim=-1)  # Shape: [batch, num_modalities]

        # 对损失进行softmax得到准确度比例（损失越大，比例越高）
        acc_softmax = torch.softmax(acc_scores / self.temperature, dim=-1)  # Shape: [batch, num_modalities]

        # ============= 2. 计算不确定性分数 ===============
        log_softmax_modalities = [torch.log_softmax(m, dim=-1) for m in modalities]
        
        # 目标均匀分布
        target_dis = torch.full_like(log_softmax_modalities[0], self.un_target_D)
        # print(f"target_dis is {target_dis}")
        # 计算每个模态与均匀分布的KL散度作为不确定性
        kl_divs = [self.kl(log_m, target_dis).sum(dim=-1) for log_m in log_softmax_modalities]
        uncerten_scores = torch.stack(kl_divs, dim=-1)  # Shape: [batch, num_modalities]
        # print(f"kl_divs is {kl_divs}")
        # print(f"uncerten_scores is {uncerten_scores}")
        # 对不确定性分数进行softmax（KL散度越大，不确定性越高，分数越低）
        uncerten_softmax = torch.softmax(-uncerten_scores / self.temperature, dim=-1) # Shape: [batch, num_modalities]
        # print(f"uncerten_softmax is {uncerten_softmax}")

        # ================= 3. 结合准确度与不确定性 =================
        # 使用beta进行加权平均
        final_ratios = acc_softmax * self.beta + uncerten_softmax * (1 - self.beta) # Shape: [batch, num_modalities]

        # # ================= 4. 生成掩码 =================
        # # 新逻辑：在样本级别进行细粒度掩码
        # gama = self.gama

        # # 1. 对每个样本，计算其模态得分的最大和最小差异
        # max_scores, _ = torch.max(final_ratios, dim=-1)
        # min_scores, _ = torch.min(final_ratios, dim=-1)
        # diffs = max_scores - min_scores # Shape: [batch]

        # # 2. 确定哪些样本需要被掩码（差异大于等于gama）
        # samples_to_mask = diffs >= gama # Shape: [batch], boolean tensor
        # if epoch_index<=5:
        #     diffs_mean = diffs.mean()
        #     diffs_median = diffs.median()
        #     print(f"mean is {diffs_mean.item()} median is {diffs_median.item()}")
        # # 3. 生成基础的“赢家通吃”掩码
        # # 首先，为所有样本创建一个全零掩码
        # winner_mask = torch.zeros_like(final_ratios)
        # # 找到每个样本最高分的模态索引
        # max_score_indices = torch.argmax(final_ratios, dim=-1)
        # # 将最高分模态的位置设置为1
        # winner_mask.scatter_(dim=1, index=max_score_indices.unsqueeze(1), value=1)

        # # 4. 根据条件选择最终掩码
        # # 如果 `samples_to_mask` 中对应位置为True，则使用 `winner_mask`；否则，使用全1掩码
        # result = torch.where(samples_to_mask.unsqueeze(1), winner_mask, torch.ones_like(final_ratios))
        # # ================= 4. 生成掩码 =================
        # 新逻辑：仅 mask 每个样本中得分最低的模态
        gama = self.gama
        
        # 1) 样本级差异
        max_scores, _ = torch.max(final_ratios, dim=-1)
        min_scores, _ = torch.min(final_ratios, dim=-1)
        diffs = max_scores - min_scores  # [batch]
        
        # 2) 需要掩码的样本（差异 >= gama）
        samples_to_mask = diffs >= gama  # [batch] bool
        # if epoch_index<=5:
        #     diffs_mean = diffs.mean()
        #     diffs_median = diffs.median()
        #     print(f"mean is {diffs_mean.item()} median is {diffs_median.item()}")
        # 3) 生成仅屏蔽最低分模态的掩码
        # 先做全 1，再把最低分索引位置置 0
        min_score_indices = torch.argmin(final_ratios, dim=-1)  # [batch]
        one_hot_min = torch.zeros_like(final_ratios)
        one_hot_min.scatter_(dim=1, index=min_score_indices.unsqueeze(1), src=torch.ones_like(min_score_indices, dtype=final_ratios.dtype).unsqueeze(1))
        loser_mask = torch.ones_like(final_ratios) - one_hot_min  # 该掩码仅在最低分位置为 0，其余为 1
        
        # 4) 条件应用掩码：满足条件用 loser_mask，否则全 1
        result = torch.where(samples_to_mask.unsqueeze(1), loser_mask, torch.ones_like(final_ratios))
        # ======================= 5. 日志输出 =====================
        if epoch_index == 0:
            tensors_to_print = {"labels": labels.squeeze()}
            metrics_info = [
                ("Prob", lambda t, i, lbl: t[i, lbl].item()),
                ("Loss", lambda t, i, _: t[i].item()),
                ("AccR", lambda t, i, _: t[i].item()),
                ("UncR", lambda t, i, _: t[i].item()),
                ("FinalR", lambda t, i, _: t[i].item()),
            ]
            modal_metrics = {}
            for idx, modality in enumerate(modalities):
                mod_name = self.model_name[idx]
                modal_metrics.setdefault(mod_name, {})
                # print(f"modality shape is {modality.shape} for {mod_name}")
                modal_metrics[mod_name]["Prob"] = torch.softmax(modality, dim=1)
                modal_metrics[mod_name]["Loss"] = acc_scores[:, idx]
                modal_metrics[mod_name]["AccR"] = acc_softmax[:, idx]
                modal_metrics[mod_name]["UncR"] = uncerten_softmax[:, idx]
                modal_metrics[mod_name]["FinalR"] = final_ratios[:, idx]

            print("\n" + "=" * 200)
            print(f"{'Epoch 0 Detailed Statistics':^200}")
            print("=" * 200)
            header = f"{'Sample':<6} | {'Label':<6} | "
            for metric_name, _ in metrics_info:
                for mod_name in self.model_name:
                    header += f"{mod_name+'_'+metric_name:<12} | "
            header += f"{'Decision':<20}"
            print(header)
            print("-" * (len(header) + 4))

            for i in range(batch_size):
                label_val = int(tensors_to_print['labels'][i].item())
                row = f"{i:<6} | {label_val:<6} | "
                for metric_name, extractor in metrics_info:
                    for mod_name in self.model_name:
                        tensor = modal_metrics[mod_name][metric_name]
                        value = extractor(
                            tensor,
                            i,
                            label_val if metric_name == "Prob" else None
                        )
                        row += f"{value:<12.4f} | "
                masked_modalities = [
                    self.model_name[k] for k, masked in enumerate(result[i]) if masked == 0
                ]
                decision = "None" if not masked_modalities else ", ".join(masked_modalities)
                row += f"{decision:<20}"
                print(row)

            print("\n" + "=" * (len(header) + 4))
        return result.bool(), final_ratios
    
#================Shapley 计算==================

class Shapley(nn.Module):
    def __init__(self, args):
        super(Shapley, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, *modalities, fusion_logits=None, labels=None, epoch_index=-1, epoch=-1):

        if not modalities:
            raise ValueError("At least one modality must be provided.")

        num_modalities = len(modalities)
        batch_size = modalities[0].shape[0]
        device = modalities[0].device
        eps = 1e-8
        m1_Mask =torch.ones(batch_size, device=device)
        m2_Mask =torch.ones(batch_size, device=device)
        m3_Mask =torch.ones(batch_size, device=device)

        contribution = {}
         # 计算softmax概率
        # 累计贡献度
        if num_modalities == 2:
            cona, conv = 0.0, 0.0
        elif num_modalities == 3:
            cona, conv, cont = 0.0, 0.0, 0.0

        prediction = self.softmax(fusion_logits)
        modal_preds = [self.softmax(logits) for logits in modalities]
        
        # 计算每个样本的 Shapley 值
        for i, label in enumerate(labels):
            fusion_pred = torch.argmax(prediction[i]).item()
            modal_pred = [torch.argmax(modal_pred[i]).item() for modal_pred in modal_preds]

            # 价值函数：预测正确为1，错误为0
            value_all = 1.0 if fusion_pred == label.item() else 0.0
            modal_values = [1.0 if idx == label.item() else 0.0 for idx in modal_pred]
            value_empty = 0.0  # 空集的价值为0
            
            if num_modalities == 2:
                # 两模态 Shapley 值计算
                # φ_m1 = 1/2 * [v({m1}) - v(∅)] + 1/2 * [v({m1,m2}) - v({m2})]
                # φ_m2 = 1/2 * [v({m2}) - v(∅)] + 1/2 * [v({m1,m2}) - v({m1})]
                
                shapley_m1 = 0.5 * (modal_values[0] - value_empty) + 0.5 * (value_all - modal_values[1])
                shapley_m2 = 0.5 * (modal_values[1] - value_empty) + 0.5 * (value_all - modal_values[0])
                # if i == 0 and epoch_index>10:
                #     print(f"shapley_m1: {shapley_m1}, shapley_m2:{shapley_m2}")
                # # 累计 Shapley 值用于统计
                # cona += shapley_m1
                # conv += shapley_m2
                
                # 基于 Shapley 值决定 mask
                # # Shapley 值越大，说明该模态贡献越大，应该 mask 掉（让弱模态得到更多训练）
                # if shapley_m1 > shapley_m2:
                #     m1_Mask[i] = 0.0
                # elif shapley_m1 < shapley_m2:
                #     m2_Mask[i] = 0.0
                # Shapley 值越大，说明该模态贡献越大，应该 mask 掉（让弱模态得到更多训练）
                if shapley_m1 < shapley_m2:
                    m1_Mask[i] = 0.0
                elif shapley_m1 > shapley_m2:
                    m2_Mask[i] = 0.0   
            elif num_modalities == 3:
                # 三模态 Shapley 值计算（需要计算所有子集）
                # 为简化，这里使用两两组合的价值
                # 注意：完整的 Shapley 值需要计算所有 2^3 = 8 个子集
                
                # 假设我们只有单模态和全模态的输出
                # 简化版 Shapley 值估计：
                shapley_m1 = (modal_values[0] - value_empty) / 3.0 + (value_all - (modal_values[1] + modal_values[2]) / 2.0) / 3.0
                shapley_m2 = (modal_values[1] - value_empty) / 3.0 + (value_all - (modal_values[0] + modal_values[2]) / 2.0) / 3.0
                shapley_m3 = (modal_values[2] - value_empty) / 3.0 + (value_all - (modal_values[0] + modal_values[1]) / 2.0) / 3.0
                
                # # 累计 Shapley 值用于统计
                # cont += shapley_m1
                # conv += shapley_m2
                # cona += shapley_m3
                
                # 找出 Shapley 值最大的模态并 mask（让弱模态得到更多训练）
                shapley_values = [shapley_m1, shapley_m2, shapley_m3]
                max_idx = shapley_values.index(max(shapley_values))
                
                if max_idx == 0:
                    m1_Mask[i] = 0.0
                elif max_idx == 1:
                    m2_Mask[i] = 0.0
                else:
                    m3_Mask[i] = 0.0
        
        # 返回 mask，根据模态数量返回对应数量的 mask
        if num_modalities == 2:
            return m1_Mask.bool(), m2_Mask.bool()
        elif num_modalities == 3:
            return m1_Mask.bool(), m2_Mask.bool(), m3_Mask.bool()
        else:
            # 对于其他情况，返回所有 mask
            return m1_Mask, m2_Mask, m3_Mask