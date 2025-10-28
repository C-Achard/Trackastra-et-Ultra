<div align="center">

<p align="center">
<img src="res/logo/logo_dark_w_bg.png" style="width:45%;"/>
</p>


![Python](https://img.shields.io/pypi/pyversions/trackastra)
[![License](https://img.shields.io/github/license/weigertlab/trackastra)](https://github.com/weigertlab/trackastra/blob/main/LICENSE)

</div>


# *Trackastra Et Ultra* - Tracking by Association with Transformers and Foundation Models Features


*Trackastra et Ultra* (`trackastra_pretrained_feats`) is a Python library that provides a small API to extract pre-trained features for use in [Trackastra](https://github.com/weigertlab/trackastra), meant for enhancing cell tracking performance using features from foundation models such as [SAM2.1](https://github.com/facebookresearch/sam2) or [CoTracker3](https://github.com/facebookresearch/co-tracker).

<p align="center">
<img src="res/model.png" alt="Updated model with pre-trained features" style="width:85%;"/>
</p>

Must be installed in order to use the SAM2.1-powered (or other arbitrary model) pre-trained model(s) in Trackastra.

## Installation

Can be installed as an optional module of Trackastra:

```bash
pip install trackastra[etultra]
```

For standalone installation (Trackastra still required):

```bash
pip install trackastra
pip install git+https://github.com/C-Achard/Trackastra-et-Ultra.git
```

## Example usage

Below is a minimal example of how to use Trackastra with pre-trained features from SAM 2.1 to track bacteria.

You may also run the [example Jupyter notebook](https://github.com/C-Achard/Trackastra-et-Ultra/blob/main/example/basic_use.ipynb) for a quick demo.

The model used here is `SAM21_general_2d`, which is also available in the [napari-trackastra plugin](https://github.com/weigertlab/napari-trackastra/).

```python
from pathlib import Path
import torch

from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks, write_to_geff
from trackastra.data import example_data_bacteria

imgs, masks = example_data_bacteria()
model = Trackastra.from_pretrained("SAM21_general_2d")

track_graph, masks_tracked = model.track(imgs, masks, mode="greedy")
ctc_tracks, ctc_masks = graph_to_ctc(
    track_graph,
    masks_tracked,
    # outdir="tracked_ctc",
)
napari_tracks, napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)
import napari
v = napari.Viewer()
v.add_image(imgs)
v.add_labels(ctc_masks)
v.add_tracks(data=napari_tracks, graph=napari_tracks_graph)
```

See the official [Trackastra documentation](https://github.com/weigertlab/trackastra) for more details on how to use the Trackastra models and API.

## Reference

If you use this code, please cite the following publications:

[Trackastra: Transformer-based cell tracking for live-cell microscopy](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf)

```
@inproceedings{gallusser2024trackastra,
  title={Trackastra: Transformer-based cell tracking for live-cell microscopy},
  author={Gallusser, Benjamin and Weigert, Martin},
  booktitle={European conference on computer vision},
  pages={467--484},
  year={2024},
  organization={Springer}
}
```

[SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)

```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
(See [the SAM2.1 repository](https://github.com/facebookresearch/sam2) for the official citation instructions.)

## Training models

To train your own pre-trained features models, please check out the [main repo fork used for training](https://github.com/C-Achard/trackastra/tree/cy/aug-zarr-caching).

Feel free to reach out for help if you'd like to train your own models, as it is still experimental.

## Authors

Made by Cyril Achard, under the supervision of Martin Weigert and with help from Benjamin Gallusser.