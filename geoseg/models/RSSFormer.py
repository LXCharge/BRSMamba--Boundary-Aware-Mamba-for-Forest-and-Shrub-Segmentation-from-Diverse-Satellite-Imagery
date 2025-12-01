import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SimpleFusion(nn.Module):
    def __init__(self, in_channels):
        super(SimpleFusion, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

    def forward(self, feat_list):
        x0 = feat_list[0]
        x0_h, x0_w = x0.size(2), x0.size(3)
        x1 = F.interpolate(
            feat_list[1], size=(x0_h, x0_w), mode="bilinear", align_corners=True
        )
        x2 = F.interpolate(
            feat_list[2], size=(x0_h, x0_w), mode="bilinear", align_corners=True
        )
        x3 = F.interpolate(
            feat_list[3], size=(x0_h, x0_w), mode="bilinear", align_corners=True
        )
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.fuse_conv(x)
        return x, x0


def calculate_variance_term(pred, gt, n_objects, delta_v, norm=2):
    """pred: bs, height * width, n_filters
    gt: bs, height * width, n_instances
    means: bs, n_instances, n_filters"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    # bs, n_loc, n_instances, n_filters
    # means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    pred = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_filters)

    # _var = (torch.clamp(torch.norm((pred - means), norm, 3) -
    #                     delta_v, min=0.0) ** 2) * gt[:, :, :, 0]
    _var = (torch.clamp(torch.norm((pred - gt), norm, 3), min=0.0)) * gt[:, :, :, 0]
    _var = torch.sum(_var)
    print("_var", _var.shape)
    var_term = 0.0
    # for i in range(bs):
    #     _var_sample = _var[i, :, :n_objects[i]]  # n_loc, n_objects
    #     _gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects
    #
    #     var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
    var_term = _var  # / bs

    return var_term


def discriminative_loss(
    input, target, n_objects, max_n_objects, delta_v, delta_d, norm, usegpu=True
):
    """input: bs, n_filters, fmap, fmap
    target: bs, n_instances, fmap, fmap
    n_objects: bs"""

    # print(input.shape)
    # print(target.shape)
    # torch.Size([8, 7, 512, 512])
    # torch.Size([8, 512, 512])

    delta_unlabeled = 1.5
    bs, n_filters, height, width = input.size()
    n_instances = target.size(1)

    input = input.permute(0, 2, 3, 1).contiguous().view(bs, height * width, n_filters)
    target = (
        target.permute(0, 2, 3, 1).contiguous().view(bs, height * width, n_instances)
    )

    var_term = calculate_variance_term(input, target, n_objects, delta_v, norm)

    unlabeled_push_weight = 1.0

    # unlabeled_push_term = _compute_unlabeled_push2(var_term,cluster_means, input, target,norm,delta_unlabeled)
    loss = unlabeled_push_weight * var_term  # * unlabeled_push_term
    print("dis_loss", loss)
    return loss


class DiscriminativeLoss(_Loss):

    def __init__(
        self, delta_var, delta_dist, norm, size_average=True, reduce=True, usegpu=True
    ):
        super(DiscriminativeLoss, self).__init__(size_average)
        self.reduce = reduce

        # assert self.size_average
        # assert self.reduce

        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.norm = int(norm)
        self.usegpu = usegpu

        assert self.norm in [1, 2]

    def forward(self, input, target, n_objects, max_n_objects):
        # _assert_no_grad(target)
        print("-------------------------------------")
        print("-------------------------------------")
        return discriminative_loss(
            input,
            target,
            n_objects,
            max_n_objects,
            self.delta_var,
            self.delta_dist,
            self.norm,
            self.usegpu,
        )


def binary_cross_entropy(pred, label, weight=None, reduction="mean", avg_factor=None):
    """Calculate the binary CrossEntropy loss with logits.

    Args:
        pred (torch.Tensor): The prediction with shape (N, *).
        label (torch.Tensor): The gt label with shape (N, *).
        weight (torch.Tensor, optional): Element-wise weight of loss with shape
             (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
             is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert pred.dim() == label.dim()

    loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

    # apply weights and do the reduction
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def soft_cross_entropy(pred, label, weight=None, reduction="mean", avg_factor=None):
    """Calculate the Soft CrossEntropy loss. The label can be float.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = -label * F.log_softmax(pred, dim=-1)
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss


def cross_entropy(pred, label, weight=None, reduction="mean", avg_factor=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction="none")

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, sigmoid=False, softmax=False, reduction="mean", loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = sigmoid
        self.use_soft = softmax
        assert not (
            self.use_soft and self.use_sigmoid
        ), "use_sigmoid and use_soft could not be set simultaneously"

        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_soft:
            self.cls_criterion = soft_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        n_pred_ch, n_target_ch = cls_score.shape[1], label.shape[1]
        if n_pred_ch == n_target_ch:
            label = torch.argmax(label, dim=1)
        else:
            label = torch.squeeze(label, dim=1)
        label = label.long()

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss_cls


loss_consis = torch.nn.L1Loss()


class MCTransAuxLoss(CrossEntropyLoss):
    def __init__(self, **kwargs):
        super(MCTransAuxLoss, self).__init__(**kwargs)

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        # print('input cls_score',cls_score.shape)
        # print('input label',label.shape)
        # input
        # cls_score
        # torch.Size([8, 7])
        # input
        # label
        # torch.Size([8, 512, 512])

        # print('input label', label)
        # To one hot
        num_classes = cls_score.shape[1]
        one_hot = []
        for l in label:
            # print('input l', torch.unique(l).shape)
            one_hot.append(
                self.one_hot(torch.unique(l), num_classes=num_classes).sum(dim=0)
            )
        label = torch.stack(one_hot)

        reduction = reduction_override if reduction_override else self.reduction
        # print('out cls_score',cls_score.shape)torch.Size([8, 7])
        # print('out cls_score',cls_score)
        # print('out label',label.shape)  torch.Size([8, 7])
        # print('out label',label)
        # loss_cls1 = loss_consis(cls_score, label)

        loss_cls1 = 1 / (1 + torch.exp(abs(cls_score - label)))
        loss_cls1 = loss_cls1.sum(1) / (2 * loss_cls1.shape[0])
        # print(loss_cls1.shape)
        # print(loss_cls1)
        # loss_cls1 = loss_consis(cls_score, label)
        loss_cls = 0
        # loss_cls = self.cls_criterion(
        #     cls_score,
        #     label,
        #     weight,
        #     reduction=reduction,
        #     avg_factor=avg_factor,
        #     **kwargs)
        # print('out loss_cls', loss_cls)
        return 0.5 * loss_cls, loss_cls1  # .cuda()#.softmax(dim=0)

    def one_hot(self, input, num_classes, dtype=torch.float):
        assert input.dim() > 0, "input should have dim of 1 or more."

        # if 1D, add singelton dim at the end
        if input.dim() == 1:
            input = input.view(-1, 1)

        sh = list(input.shape)

        assert sh[1] == 1, "labels should have a channel with length equals to one."
        sh[1] = num_classes

        o = torch.zeros(size=sh, dtype=dtype, device=input.device)
        labels = o.scatter_(dim=1, index=input.long(), value=1)

        return labels


def softmax_focalloss(y_pred, y_true, ignore_index=-1, gamma=2.0, normalize=False):
    """
    Args:
        y_pred: [N, #class, H, W]
        y_true: [N, H, W] from 0 to #class
        gamma: scalar
    Returns:
    """

    y_pred = y_pred.unsqueeze(dim=1)
    y_true = y_true.unsqueeze(dim=1)
    print("y_pred", y_pred.shape)
    print("y_true", y_true.shape)
    losses = F.cross_entropy(
        y_pred, y_true, ignore_index=ignore_index, reduction="none"
    )
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        print(masked_y_true.shape)  # torch.Size([8, 7])
        print(masked_y_true.shape)
        # print(masked_y_true.reshpe(y_pred.shape[0],y_pred.shape[1],1,1).shape)
        # print(modulating_factor.reshpe(y_pred.shape[0],1,1,1).shape)

        # print(masked_y_true.unsqueeze(dim=2).unsqueeze(dim=3).shape)
        # print(modulating_factor.unsqueeze(dim=2).unsqueeze(dim=3).shape)
        modulating_factor = torch.gather(
            modulating_factor.unsqueeze(dim=1),
            dim=1,
            index=masked_y_true.to(torch.int64).unsqueeze(dim=1),
        ).squeeze_(dim=1)
        scale = 1.0
        if normalize:
            scale = losses.sum() / (losses * modulating_factor).sum()
    losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

    return losses


def _masked_ignore(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int = -1):
    # usually used for BCE-like losses
    y_pred = y_pred.reshape((-1,))
    y_true = y_true.reshape((-1,))
    valid = y_true != ignore_index
    y_true = y_true.masked_select(valid).float()
    y_pred = y_pred.masked_select(valid).float()
    return y_pred, y_true


def binary_cross_entropy_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -1,
):
    output, target = _masked_ignore(output, target, ignore_index)
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)


def tversky_loss_with_logits(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha: float,
    beta: float,
    smooth_value: float = 1.0,
    ignore_index: int = -1,
):
    y_pred, y_true = _masked_ignore(y_pred, y_true, ignore_index)

    y_pred = y_pred.sigmoid()
    tp = (y_pred * y_true).sum()
    # fp = (y_pred * (1 - y_true)).sum()
    fp = y_pred.sum() - tp
    # fn = ((1 - y_pred) * y_true).sum()
    fn = y_true.sum() - tp

    tversky_coeff = (tp + smooth_value) / (tp + alpha * fn + beta * fp + smooth_value)
    return 1.0 - tversky_coeff


def dice_coeff(
    y_pred,
    y_true,
    weights: torch.Tensor,
    smooth_value: float = 1.0,
):
    y_pred = y_pred[:, weights]
    y_true = y_true[:, weights]
    inter = torch.sum(y_pred * y_true, dim=0)
    z = y_pred.sum(dim=0) + y_true.sum(dim=0) + smooth_value

    return ((2 * inter + smooth_value) / z).mean()


def select(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    assert y_pred.ndim == 4 and y_true.ndim == 3
    c = y_pred.size(1)
    y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, c)
    y_true = y_true.reshape(-1)

    valid = y_true != ignore_index

    y_pred = y_pred[valid, :]
    y_true = y_true[valid]
    return y_pred, y_true


def dice_loss_with_logits(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    smooth_value: float = 1.0,
    ignore_index: int = -1,
    ignore_channel: int = -1,
):
    c = y_pred.size(1)
    y_pred, y_true = select(y_pred, y_true, ignore_index)
    weight = torch.as_tensor([True] * c, device=y_pred.device)
    if c == 1:
        y_prob = y_pred.sigmoid()
        return 1.0 - dice_coeff(y_prob, y_true.reshape(-1, 1), weight, smooth_value)
    else:
        y_prob = y_pred.softmax(dim=1)
        y_true = F.one_hot(y_true, num_classes=c)
        if ignore_channel != -1:
            weight[ignore_channel] = False

        return 1.0 - dice_coeff(y_prob, y_true, weight, smooth_value)


class SegmentationLoss(nn.Module):
    def __init__(self, loss_config):
        super(SegmentationLoss, self).__init__()
        self.loss_config = loss_config

        self.criterion_discriminative = DiscriminativeLoss(
            delta_var=0.5, delta_dist=2, norm=2
        )
        self.criterion_aux = MCTransAuxLoss()
        print("SegmentationLossaux")

    def forward(self, y_pred, y_true: torch.Tensor, y_pred2):
        loss_dict = dict()
        # print('dis_loss')
        # mem = torch.cuda.max_memory_allocated() // 1024 // 1024
        # loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(y_pred.device)
        if "ce" in self.loss_config:
            # loss_ce = F.cross_entropy(y_pred, y_true.long(), ignore_index=-1)
            # print('loss_ce', loss_ce)

            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(
                y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb)
            )
            bg_y_true[invalidmask] = 0
            # y_pred, bg_y_true = _masked_ignore(y_predb, bg_y_true, ignore_index=-1)
            # loss_dict['ce_loss'] = loss_ce

            # loss_dict['aux3_loss'],l1= self.criterion_aux(y_pred2, bg_y_true)

            loss, l1 = self.criterion_aux(y_pred2, bg_y_true)
            # print('l1',l1)
            # similarity = torch.cosine_similarity(y_pred2, bg_y_true, dim=0)
            # similarity = torch.mean(similarity)
            # l_S = 1 - similarity
            result = l1  # round(float(l1), 1)
            # print(result)
            # loss_dict['ce_loss'] = abs(result)*loss_ce
            loss_dict["fc_loss"] = softmax_focalloss(y_pred, y_true, gamma=result)
            # print(y_pred.shape)
            # print(y_true.shape)
            # torch.Size([8, 7, 512, 512])
            # torch.Size([8, 512, 512])
            # y_predb = y_pred[:, 0, :, :]
            # print(y_predb.shape)
            # invalidmask = y_true == -1
            # bg_y_true = torch.where(y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            # print(bg_y_true.shape)
            # bg_y_true = y_pred
            # for i in range(7):
            #     bg_y_true[:, i, :, :] = torch.where(y_true == i, i*torch.ones_like(y_true), torch.zeros_like(y_true))
            # loss_consis = torch.nn.L1Loss()
            # l1 = loss_consis(y_pred2, bg_y_true)

            # result = round(float(l1), 1)

        if "fcloss" in self.loss_config:
            loss_dict["fc_loss"] = softmax_focalloss(
                y_pred, y_true, gamma=self.loss_config.fcloss.gamma, normalize=True
            )

        if "bceloss" in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(
                y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb)
            )
            bg_y_true[invalidmask] = -1
            loss_dict["bceloss"] = (
                binary_cross_entropy_with_logits(y_predb, bg_y_true, ignore_index=-1)
                * self.loss_config.bceloss.scaler
            )

        if "tverloss" in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(
                y_true > 0, torch.ones_like(y_predb), torch.zeros_like(y_predb)
            )
            bg_y_true[invalidmask] = -1
            loss_dict["tverloss"] = (
                tversky_loss_with_logits(
                    y_predb,
                    bg_y_true,
                    self.loss_config.tverloss.alpha,
                    self.loss_config.tverloss.beta,
                    ignore_index=-1,
                )
                * self.loss_config.tverloss.scaler
            )

        if "diceloss" in self.loss_config:

            loss_dict["dice_loss"] = (
                dice_loss_with_logits(y_pred, y_true) * self.loss_config.diceloss.scaler
            )
            # loss_dict['ce_loss'] = F.cross_entropy(y_pred, y_true.long(), ignore_index=-1)

        return loss_dict


class RSSFormer(nn.Module):

    def __init__(
        self,
        backbone_name="efficientnet_b3",
        pretrained=True,
        num_classes=7,
        upsample_scale=4.0,
    ):
        super(RSSFormer, self).__init__()
        self.config = dict(
            neck=dict(
                in_channels=720,
            ),
            classes=7,
            head=dict(
                in_channels=720,
                upsample_scale=4.0,
            ),
            loss=dict(
                ce=dict(),
            ),
        )
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            output_stride=32,
            out_indices=(1, 2, 3, 4),
            pretrained=pretrained,
        )
        neck_in_channels = sum(self.backbone.feature_info.channels())
        self.neck = SimpleFusion(neck_in_channels)
        self.head = nn.Sequential(
            # nn.Conv2d(self.config.head.in_channels, self.config.classes, 1),
            nn.Conv2d(neck_in_channels, num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=upsample_scale),
        )
        # self.loss = SegmentationLoss(self.config.loss)

        # self.headaux = nn.Sequential(nn.Linear(480, 128),
        #                           nn.Linear(128, 7))
        self.headaux = nn.Sequential(nn.Linear(32, 7))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, y=None):
        pred_list = self.backbone(x)
        # print('the flops is G=====================================')
        # from thop import profile
        # input = torch.randn(1, 3, 224, 224).cuda()
        # flops, params = profile(self.backbone, inputs=(input,))
        # print('the flops is {}G,the params is {}M'.format(round(flops / (10 ** 9), 2), round(params / (10 ** 6), 2)))
        # logit = self.neck(pred_list)
        logit, f0 = self.neck(pred_list)
        # print('logit',logit.shape)    #torch.Size([16, 480, 128, 128])
        x = self.avg_pool(f0)
        # print('x', x.flatten(1).shape)
        logit = self.head(logit)
        # print('logit', logit.shape) torch.Size([16, 7, 512, 512])
        return logit.softmax(dim=1)


if __name__ == "__main__":
    model = RSSFormer().to("cuda")
    input = torch.randn(1, 3, 512, 512).to("cuda")
    output = model(input)
    print(output.shape)
