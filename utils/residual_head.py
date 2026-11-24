import torch
from torch import nn

from wan.modules.model import WanModel


class ResidualHead(nn.Module):
    """Repeated Residual Adaptation Head (RRAH).

    A lightweight Wan DiT (``num_layers`` blocks) that takes the predicted velocity
    (optionally concatenated with the LQ latent) and predicts a residual correction.
    It shares all hyper-parameters except depth / input channels with ``base_model``.
    """

    def __init__(self, in_channels, base_model, num_layers=1, out_channels=16):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if base_model is None:
            raise ValueError("base_model must be provided for ResidualHead DiT")

        self.model_type = base_model.model_type
        self.dit = WanModel(
            model_type=base_model.model_type,
            patch_size=base_model.patch_size,
            text_len=base_model.text_len,
            in_dim=in_channels,
            dim=base_model.dim,
            ffn_dim=base_model.ffn_dim,
            freq_dim=base_model.freq_dim,
            text_dim=base_model.text_dim,
            out_dim=out_channels,
            num_heads=base_model.num_heads,
            num_layers=num_layers,
            window_size=base_model.window_size,
            qk_norm=base_model.qk_norm,
            cross_attn_norm=base_model.cross_attn_norm,
            eps=base_model.eps,
        )

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        squeeze = False
        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze = True

        if isinstance(context, torch.Tensor):
            context_list = [context[i] for i in range(context.shape[0])]
        else:
            context_list = context

        target_dtype = self.dit.patch_embedding.weight.dtype
        target_dtype_dit = self.dit.text_embedding[0].weight.dtype
        x_list = [u.to(dtype=target_dtype) for u in x]

        if self.model_type == "i2v":
            if clip_fea is None:
                clip_fea = x.new_zeros((x.size(0), 257, 1280))
            if y is None:
                y = [torch.zeros_like(u) for u in x_list]

        if isinstance(context_list, (list, tuple)):
            context_list = [c.to(dtype=target_dtype_dit) for c in context_list]
        else:
            context_list = context_list.to(dtype=target_dtype_dit)
        if clip_fea is not None:
            clip_fea = clip_fea.to(dtype=target_dtype)
        if y is not None:
            y = [u.to(dtype=target_dtype) for u in y]

        out_list = self.dit(
            x=x_list,
            t=t,
            context=context_list,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y,
        )
        out = torch.stack(out_list, dim=0)
        return out[0] if squeeze else out


def residual_in_channels(residual_input):
    if residual_input in ("pred", "control"):
        return 16
    if residual_input == "pred+control":
        return 32
    raise ValueError(f"Unknown residual_input: {residual_input}")
