import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class Adapter(nn.Module):
    def __init__(
        self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_E = nn.Parameter(self.weight.new_zeros(r, 1))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.ranknum = nn.Parameter(self.weight.new_zeros(1), requires_grad=False)
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha > 0 else float(self.r)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A,B the same way as the default for nn.Linear
            # and E (singular values) for zero
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (
                    T(self.lora_B @ (self.lora_A * self.lora_E))
                    * self.scaling
                    / (self.ranknum + 1e-5)
                )
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (
                    T(self.lora_B @ (self.lora_A * self.lora_E))
                    * self.scaling
                    / (self.ranknum + 1e-5)
                )
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (
                    (
                        self.lora_dropout(x)
                        @ (self.lora_A * self.lora_E).T
                        @ self.lora_B.T
                    )
                    * self.scaling
                    / (self.ranknum + 1e-5)
                )
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class RankAllocator(object):
    """
    The RankAllocator for AdaLoRA Model that will be called every training step.
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model: the model that we apply AdaLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank.
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter.
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter.
    """

    def __init__(
        self,
        model,
        lora_r: int,
        target_rank: int,
        init_warmup: int,
        final_warmup: int,
        mask_interval: int,
        beta1: float,
        beta2: float,
        total_step: Optional[int] = None,
        target_total_rank: Optional[int] = None,
        tb_writter=None,
        tb_writter_loginterval: int = 500,
    ):
        self.ave_target_rank = target_rank
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {}
        self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval

        assert self.beta1 < 1 and self.beta1 > 0
        assert self.beta2 < 1 and self.beta2 > 0

    def set_total_step(self, total_step: int):
        # Set total step number
        self.total_step = total_step
        assert self.total_step > self.initial_warmup + self.final_warmup

    def get_rank_pattern(self):
        # Return rank pattern
        return self.rank_pattern

    def get_lora_param_name(self):
        # Prepare the budget scheduler
        self.name_set = set()
        self.total_rank = 0
        self.shape_dict = {}
        for n, p in self.model.named_parameters():
            if "lora_A" in n:
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0)
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set))
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set)

    def schedule_threshold(self, step: int):
        # Global budget schedule
        mask_ind = False
        target_rank = self.target_rank
        initial_warmup = self.initial_warmup
        final_warmup = self.final_warmup
        total_step = self.total_step
        self.global_step = step
        if step <= initial_warmup:
            # Initial warmup
            curr_rank = self.total_rank
            mask_ind = False
        elif step > total_step - final_warmup:
            # Final fine-tuning
            curr_rank = self.target_rank
            # Fix the rank pattern by
            # always masking the same unimportant singluar values
            mask_ind = True
        else:
            # Budget decreasing
            mul_coeff = 1 - (step - initial_warmup) / (
                total_step - final_warmup - initial_warmup
            )
            curr_rank = target_rank + (self.total_rank - target_rank) * (mul_coeff**3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False
        return curr_rank, mask_ind

    def update_ipt(self, model):
        for n, p in model.named_parameters():
            if "lora_" in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    # Calculate sensitivity
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Update sensitivity
                    self.exp_avg_ipt[n] = (
                        self.beta1 * self.exp_avg_ipt[n]
                        + (1 - self.beta1) * self.ipt[n]
                    )
                    # Update uncertainty
                    self.exp_avg_unc[n] = (
                        self.beta2 * self.exp_avg_unc[n]
                        + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    )

    def calculate_score(self, n, p=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone()
        else:
            raise ValueError("Unexcptected Metric: %s" % metric)
        return ipt_score

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_target_rank(self, model, curr_rank):
        is_dict = {}
        combine_dict = {}
        singular_dict = {}
        # Calculate the importance score for each sub matrix
        for n, p in model.named_parameters():
            if "lora_A" in n:
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                if name_mat not in combine_dict:
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_B" in n:
                hdim_b, rdim = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                if name_mat not in combine_dict:
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_E" in n:
                ipt_score = self.calculate_score(n, p=p, metric="ipt")
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score

        # Combine the importance scores
        all_is = []
        for name_mat in combine_dict:
            ipt_E = singular_dict[name_mat]
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat % "lora_E"
            is_dict[name_E] = sum_ipt.view(-1, 1)
            all_is.append(sum_ipt.view(-1))

        # Calculate the masking threshold
        mask_threshold = torch.kthvalue(
            torch.cat(all_is), (self.total_rank - curr_rank)
        )[0].item()

        # Mask out unimportant singular values
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            for n, p in model.named_parameters():
                if "lora_E" in n:
                    p.data.masked_fill_(is_dict[n] <= mask_threshold, 0.0)
                    ranknum = (is_dict[n] > mask_threshold).sum().item()

                    if (
                        self.tb_writter is not None
                        and self.global_step % self.log_interval == 0
                    ):
                        self.tb_writter.add_scalar(
                            "Ranknum/%s" % (n,), ranknum, self.global_step
                        )
                        self.rank_pattern[n] = ranknum
                        curr_sum_rank += ranknum
                        sum_param += (
                            ranknum * self.shape_dict[n.replace("lora_E", "lora_A")][1]
                        )
                        sum_param += (
                            ranknum * self.shape_dict[n.replace("lora_E", "lora_B")][0]
                        )

            if (
                self.tb_writter is not None
                and self.global_step % self.log_interval == 0
            ):
                self.tb_writter.add_scalar(
                    "Budget/total_rank", curr_sum_rank, self.global_step
                )
                self.tb_writter.add_scalar(
                    "Budget/mask_threshold", mask_threshold, self.global_step
                )
                self.tb_writter.add_scalar(
                    "Budget/sum_param", sum_param, self.global_step
                )

        return mask_threshold

    def update_and_mask(self, model, global_step):
        if global_step < self.total_step - self.final_warmup:
            # Update importance scores element-wise
            self.update_ipt(model)
            # do not update ipt during final fine-tuning
        # Budget schedule
        curr_rank, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            # Mask to target budget
            mask_threshold = self.mask_to_target_rank(model, curr_rank)
        else:
            mask_threshold = None
        self._maybe_tb_writter_log(model)
        return curr_rank, mask_threshold

    def _maybe_tb_writter_log(self, model):
        if self.tb_writter is not None and self.global_step % self.log_interval == 0:
            with torch.no_grad():
                regu_loss = []
                for n, p in model.named_parameters():
                    if "lora_A" in n or "lora_B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov - I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s" % n, orth_regu.item(), self.global_step
                        )
                self.tb_writter.add_scalar(
                    "train/orth_regu_loss",
                    sum(regu_loss) / len(regu_loss),
                    self.global_step,
                )


def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`.
    regu_loss, num_param = 0.0, 0
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov - I, p="fro")
            num_param += 1
    return regu_weight * regu_loss / num_param


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_Ax = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_Ay = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_Bx = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_By = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        self.lora_wx = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lora_wy = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lora_wx.data.fill_(0.5)
        self.lora_wy.data.fill_(0.5)
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_Ax"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_Ax, a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bx)
        if hasattr(self, "lora_Ay"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_Ay, a=math.sqrt(5))
            nn.init.zeros_(self.lora_By)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            resultx = F.linear(x, T(self.weight), bias=self.bias)
            resulty = F.linear(y, T(self.weight), bias=self.bias)
            if self.r > 0:
                # multimodal LoRA
                lorax = self.lora_dropout(x) @ self.lora_Ax.T @ self.lora_Bx.T
                loray = self.lora_dropout(y) @ self.lora_Ay.T @ self.lora_By.T
                resultx += self.lora_wx * lorax + (1 - self.lora_wx) * loray
                resulty += self.lora_wy * loray + (1 - self.lora_wy) * lorax

                # # ## ablation: The standard LoRA with multimodal
                # resultx += (self.lora_dropout(x) @ self.lora_Ax.T @ self.lora_Bx.T) * self.scaling
                # resulty += (self.lora_dropout(y) @ self.lora_Ay.T @ self.lora_By.T) * self.scaling
            return resultx, resulty
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        assert (
            out_features % len(enable_lora) == 0
        ), "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_Ax = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features))
            )
            self.lora_Bx = nn.Parameter(
                self.weight.new_zeros(
                    (out_features // len(enable_lora) * sum(enable_lora), r)
                )
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.lora_Ay = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features))
            )
            self.lora_By = nn.Parameter(
                self.weight.new_zeros(
                    (out_features // len(enable_lora) * sum(enable_lora), r)
                )
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_Ax"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_Ax, a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bx)
        if hasattr(self, "lora_Ay"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_Ay, a=math.sqrt(5))
            nn.init.zeros_(self.lora_By)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            resultx = F.linear(x, T(self.weight), bias=self.bias)
            resulty = F.linear(y, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_Ax = F.linear(self.lora_dropout(x), self.lora_Ax)
                after_Bx = F.conv1d(
                    after_Ax.transpose(-2, -1),  # B, 12, M
                    self.lora_Bx.unsqueeze(-1),  # 3072, 4, 1
                    groups=sum(self.enable_lora),
                ).transpose(-2, -1)
                resultx += self.zero_pad(after_Bx) * self.scaling

                after_Ay = F.linear(self.lora_dropout(y), self.lora_Ay)
                after_By = F.conv1d(
                    after_Ay.transpose(-2, -1),  # B, 12, M
                    self.lora_By.unsqueeze(-1),  # 3072, 4, 1
                    groups=sum(self.enable_lora),
                ).transpose(-2, -1)
                resulty += self.zero_pad(after_By) * self.scaling
            return resultx, resulty
