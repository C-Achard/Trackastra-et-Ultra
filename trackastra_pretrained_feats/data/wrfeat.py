from collections import OrderedDict

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from trackastra.data.wrfeat import WRFeatures, _border_dist_fast

from ..pretrained_features import FeatureExtractor

_PROPERTIES = {
    "regionprops": (
        "area",
        "intensity_mean",
        "intensity_max",
        "intensity_min",
        "inertia_tensor",
    ),
    "regionprops2": (
        "equivalent_diameter_area",
        "intensity_mean",
        "inertia_tensor",
        "border_dist",
    ),
    "regionprops_small": (
        "area",
        "inertia_tensor",
    ),
}
DEFAULT_PROPERTIES = "regionprops2"


class WRPretrainedFeatures(WRFeatures):
    """WindowedRegion with features from pre-trained models."""

    def __init__(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        timepoints: np.ndarray,
        features: OrderedDict[np.ndarray],
        additional_properties: str | None = None,
    ):
        super().__init__(coords, labels, timepoints, features)
        self.additional_properties = additional_properties

    @property
    def features_stacked(self):
        if not self.features or (
            len(self.features) == 1 and "pretrained_feats" in self.features
        ):
            # logger.warning("No features to stack")
            return None
        feats = np.concatenate(
            [v for k, v in self.features.items() if k != "pretrained_feats"], axis=-1
        )
        # raise if any NaNs in features
        return feats

    @property
    def pretrained_feats(self):
        return super().pretrained_feats
        # if "pretrained_feats" in self.features:
        #     return self.features["pretrained_feats"]
        # return None

    @classmethod
    def from_mask_img(
        cls,
        img: np.ndarray,
        mask: np.ndarray,
        feature_extractor: FeatureExtractor,
        t_start: int = 0,
        additional_properties: str | None = None,
        # embeddings: torch.Tensor | None = None,
    ) -> "WRPretrainedFeatures":
        ndim = img.ndim - 1
        if ndim != 2:
            raise ValueError("Only 2D data is supported")

        df, coords, labels, timepoints, properties = cls.get_regionprops_features(
            additional_properties, mask, img, t_start=t_start
        )
        # if embeddings is None:
        _, features = feature_extractor.extract_embedding(
            mask, timepoints, labels, coords
        )
        # else:
        # _, features = feature_extractor.extract_embedding(mask, timepoints, labels, coords, embs=embeddings)
        features = features.detach().cpu().numpy()
        feats_dict = OrderedDict(pretrained_feats=features)
        # Add additional features similarly to WRFeatures if any
        if additional_properties is not None:
            for p in properties:
                feats_dict[p] = np.stack(
                    [
                        df[c].values.astype(np.float32)
                        for c in df.columns
                        if c.startswith(p)
                    ],
                    axis=-1,
                )

        return cls(
            coords=coords,
            labels=labels,
            timepoints=timepoints,
            features=feats_dict,
            additional_properties=additional_properties,
        )

    @staticmethod
    def get_regionprops_features(properties, mask, img, t_start=0):
        """Extracts regionprops features from a mask and image."""
        img = np.asarray(img)
        mask = np.asarray(mask)
        _ntime, ndim = mask.shape[0], mask.ndim - 1
        if ndim not in (2, 3):
            raise ValueError("Only 2D or 3D data is supported")

        if properties is None:
            properties = ()
        else:
            properties = tuple(_PROPERTIES[properties])

        if "label" in properties or "centroid" in properties:
            raise ValueError(
                f"label and centroid should not be in properties {properties}"
            )

        if "border_dist" in properties:
            use_border_dist = True
            # remove border_dist from properties
            properties = tuple(p for p in properties if p != "border_dist")
        else:
            use_border_dist = False

        df_properties = ("label", "centroid", *properties)
        dfs = []
        for i, (y, x) in enumerate(zip(mask, img)):
            _df = pd.DataFrame(
                regionprops_table(y, intensity_image=x, properties=df_properties)
            )
            _df["timepoint"] = i + t_start
            if use_border_dist:
                _df["border_dist"] = _border_dist_fast(y)

            dfs.append(_df)
        df = pd.concat(dfs)

        if use_border_dist:
            properties = (*properties, "border_dist")

        timepoints = df["timepoint"].values.astype(np.int32)
        labels = df["label"].values.astype(np.int32)
        coords = df[[f"centroid-{i}" for i in range(ndim)]].values.astype(np.float32)

        # if any NaNs in features, raise
        if df.isnull().values.any():
            raise ValueError("NaNs found in features DataFrame")

        return df, coords, labels, timepoints, properties
