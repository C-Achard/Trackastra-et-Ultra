import logging
from typing import Literal

import torch
import torch.nn as nn

from trackastra.model.model import TrackingTransformer, EncoderLayer, DecoderLayer
from trackastra.model.model_parts import (
    FeedForward,
    PositionalEncoding,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TrackingTransformerwPretrainedFeats(TrackingTransformer):
    def __init__(
        self,
        coord_dim: int = 3,
        feat_dim: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        pos_embed_per_dim: int = 32,
        feat_embed_per_dim: int = 1,
        window: int = 6,
        spatial_pos_cutoff: int = 256,
        attn_positional_bias: Literal["bias", "rope", "none"] = "rope",
        attn_positional_bias_n_spatial: int = 16,
        causal_norm: Literal[
            "none", "linear", "softmax", "quiet_softmax"
        ] = "quiet_softmax",
        attn_dist_mode: str = "v0",
        # Pretrained features arguments
        pretrained_feat_dim: int = 0,
        reduced_pretrained_feat_dim: int = 128,
        disable_xy_coords: bool = False,
        disable_all_coords: bool = False,
    ):
        super().__init__(
            coord_dim=coord_dim,
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            pos_embed_per_dim=pos_embed_per_dim,
            feat_embed_per_dim=feat_embed_per_dim,
            window=window,
            spatial_pos_cutoff=spatial_pos_cutoff,
            attn_positional_bias=attn_positional_bias,
            attn_positional_bias_n_spatial=attn_positional_bias_n_spatial,
            causal_norm=causal_norm,
            attn_dist_mode=attn_dist_mode,
        )

        self.config = dict(
            coord_dim=coord_dim,
            feat_dim=feat_dim,
            pretrained_feat_dim=pretrained_feat_dim,
            reduced_pretrained_feat_dim=reduced_pretrained_feat_dim,
            pos_embed_per_dim=pos_embed_per_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            window=window,
            dropout=dropout,
            attn_positional_bias=attn_positional_bias,
            attn_positional_bias_n_spatial=attn_positional_bias_n_spatial,
            spatial_pos_cutoff=spatial_pos_cutoff,
            feat_embed_per_dim=feat_embed_per_dim,
            causal_norm=causal_norm,
            attn_dist_mode=attn_dist_mode,
            disable_xy_coords=disable_xy_coords,
            disable_all_coords=disable_all_coords,
        )

        if pretrained_feat_dim > 0:
            self.reduced_pretrained_feat_dim = reduced_pretrained_feat_dim
        else:
            self.reduced_pretrained_feat_dim = 0
        self._return_norms = True
        self.norms = {}

        self._disable_xy_coords = disable_xy_coords
        self._disable_all_coords = disable_all_coords

        if self._disable_all_coords:
            coords_proj_dims = 0
        elif self._disable_xy_coords:
            coords_proj_dims = pos_embed_per_dim
        else:
            coords_proj_dims = (1 + coord_dim) * pos_embed_per_dim

        feats_proj_dims = feat_dim * feat_embed_per_dim

        self.proj = nn.Linear(
            coords_proj_dims + feats_proj_dims + self.reduced_pretrained_feat_dim,
            d_model,
        )
        self.norm = nn.LayerNorm(d_model)

        self.encoder = nn.ModuleList([
            EncoderLayer(
                coord_dim,
                d_model,
                nhead,
                dropout,
                window=window,
                cutoff_spatial=spatial_pos_cutoff,
                positional_bias=attn_positional_bias,
                positional_bias_n_spatial=attn_positional_bias_n_spatial,
                attn_dist_mode=attn_dist_mode,
            )
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(
                coord_dim,
                d_model,
                nhead,
                dropout,
                window=window,
                cutoff_spatial=spatial_pos_cutoff,
                positional_bias=attn_positional_bias,
                positional_bias_n_spatial=attn_positional_bias_n_spatial,
                attn_dist_mode=attn_dist_mode,
            )
            for _ in range(num_decoder_layers)
        ])

        self.head_x = FeedForward(d_model)
        self.head_y = FeedForward(d_model)

        if feat_embed_per_dim > 1:
            self.feat_embed = PositionalEncoding(
                cutoffs=(1000,) * feat_dim,
                n_pos=(feat_embed_per_dim,) * feat_dim,
                cutoffs_start=(0.01,) * feat_dim,
            )
        else:
            self.feat_embed = nn.Identity()

        if pretrained_feat_dim > 0:
            self.ptfeat_proj = nn.Sequential(
                nn.Linear(pretrained_feat_dim, self.reduced_pretrained_feat_dim),
            )
            self.ptfeat_norm = nn.LayerNorm(self.reduced_pretrained_feat_dim)
        else:
            self.ptfeat_proj = nn.Identity()
            self.ptfeat_norm = nn.Identity()

        if self._disable_all_coords:
            self.pos_embed = nn.Identity()

        elif self._disable_xy_coords:
            self.pos_embed = PositionalEncoding(
                cutoffs=(window,),
                n_pos=(pos_embed_per_dim,),
            )
        else:
            self.pos_embed = PositionalEncoding(
                cutoffs=(window,) + (spatial_pos_cutoff,) * coord_dim,
                n_pos=(pos_embed_per_dim,) * (1 + coord_dim),
            )

        # self.pos_embed = NoPositionalEncoding(d=pos_embed_per_dim * (1 + coord_dim))

    def forward(
        self, coords, features=None, pretrained_features=None, padding_mask=None
    ):
        assert coords.ndim == 3 and coords.shape[-1] in (3, 4)
        _B, _N, _D = coords.shape
        device = coords.device.type

        # disable padded coords (such that it doesnt affect minimum)
        if padding_mask is not None:
            coords = coords.clone()
            coords[padding_mask] = coords.max()

        # remove temporal offset
        min_time = coords[:, :, :1].min(dim=1, keepdims=True).values
        coords = coords - min_time

        if self._disable_xy_coords:
            coords_feat = coords[:, :, :1].clone()
        else:
            coords_feat = coords.clone()

        if not self._disable_all_coords:
            pos = self.pos_embed(coords_feat)
        else:
            pos = None

        if self._return_norms:
            self.norms = {}
            if not self._disable_all_coords:
                self.norms["pos_embed"] = pos.norm(dim=-1).detach().cpu().mean().item()
                self.norms["coords"] = (
                    coords_feat.norm(dim=-1).detach().cpu().mean().item()
                )

        with torch.amp.autocast(enabled=False, device_type=device):
            # Determine if we have any features to use
            has_features = features is not None and features.numel() > 0
            has_pretrained = (
                pretrained_features is not None
                and pretrained_features.numel() > 0
                and self.config["pretrained_feat_dim"] > 0
            )

            if self._return_norms:
                if has_features:
                    self.norms["features"] = (
                        features.norm(dim=-1).detach().cpu().mean().item()
                    )
                if has_pretrained:
                    self.norms["pretrained_features"] = (
                        pretrained_features.norm(dim=-1).detach().cpu().mean().item()
                    )

            if not has_features and not has_pretrained:
                if self._disable_all_coords:
                    raise ValueError(
                        "features is None and all coords are disabled. Please enable at least one of the two."
                    )
                features_out = pos
            else:
                # Start with features if present, else None
                features_out = self.feat_embed(features) if has_features else None
                if self._return_norms and has_features:
                    self.norms["features_out"] = (
                        features_out.norm(dim=-1).detach().cpu().mean().item()
                    )

                # Add pretrained features if configured
                if self.config["pretrained_feat_dim"] > 0 and has_pretrained:
                    pt_features = self.ptfeat_proj(pretrained_features)
                    pt_features = self.ptfeat_norm(pt_features).squeeze()
                    if self._return_norms:
                        self.norms["pt_features_out"] = (
                            pt_features.norm(dim=-1).detach().cpu().mean().item()
                        )
                    if features_out is not None:
                        try:
                            if (
                                features_out.shape[0] == 1
                                and len(pt_features.shape) == 2
                            ):
                                pt_features = pt_features.unsqueeze(0)
                            features_out = torch.cat(
                                (features_out, pt_features), dim=-1
                            )
                        except RuntimeError as e:
                            logger.error(
                                f"Pretrained features shape: {pt_features.shape}"
                            )
                            logger.error(f"Features shape: {features_out.shape}")
                            raise e
                    else:
                        features_out = pt_features

                # Add encoded coords if not disabled
                if not self._disable_all_coords:
                    if features_out is not None:
                        features_out = torch.cat((pos, features_out), axis=-1)
                    else:
                        features_out = pos

            features = self.proj(features_out)
            if self._return_norms:
                self.norms["features_cat"] = (
                    features_out.norm(dim=-1).detach().cpu().mean().item()
                )
                self.norms["features_proj"] = (
                    features.norm(dim=-1).detach().cpu().mean().item()
                )
        # Clamp input when returning to mixed precision
        features = features.clamp(
            torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
        )
        features = self.norm(features)

        x = features

        # encoder
        for enc in self.encoder:
            x = enc(x, coords=coords, padding_mask=padding_mask)

        y = features
        # decoder w cross attention
        for dec in self.decoder:
            y = dec(y, x, coords=coords, padding_mask=padding_mask)
            # y = dec(y, y, coords=coords, padding_mask=padding_mask)

        x = self.head_x(x)
        y = self.head_y(y)

        # outer product is the association matrix (logits)
        A = torch.einsum("bnd,bmd->bnm", x, y)  # /math.sqrt(_D)

        if torch.any(torch.isnan(A)):
            logger.error("NaN in A")

        return A
