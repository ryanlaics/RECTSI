import numpy as np
import torch

from . import TemporalDataset, SpatioTemporalDataset


class ImputationDataset(TemporalDataset):

    def __init__(self, data,
                 index=None,
                 mask=None,
                 eval_mask=None,
                 freq=None,
                 trend=None,
                 scaler=None,
                 window=24,
                 stride=1,
                 exogenous=None,
                 adjs=None,
                 positions=None,
                 adj_label=None):
        if mask is None:
            mask = np.ones_like(data)
        if exogenous is None:
            exogenous = dict()
        exogenous['mask_window'] = mask
        if eval_mask is not None:
            exogenous['eval_mask_window'] = eval_mask
        super(ImputationDataset, self).__init__(data,
                                                index=index,
                                                exogenous=exogenous,
                                                trend=trend,
                                                scaler=scaler,
                                                freq=freq,
                                                window=window,
                                                horizon=window,
                                                delay=-window,
                                                stride=stride,
                                                adjs=adjs,
                                                positions=positions,
                                                adj_label=adj_label)

    def get(self, item, preprocess=False):
        res, transform = super(ImputationDataset, self).get(item, preprocess)

        third_of_last_dim = res['x'].shape[-1] // 3
        transformed_part = torch.where(
            res['mask'],
            res['x'][:, :third_of_last_dim],
            torch.zeros_like(res['x'][:, :third_of_last_dim])
        )

        remaining_part = res['x'][:, third_of_last_dim:]
        res['x'] = torch.cat([transformed_part, remaining_part], dim=1)

        return res, transform


class GraphImputationDataset(ImputationDataset, SpatioTemporalDataset):
    pass
