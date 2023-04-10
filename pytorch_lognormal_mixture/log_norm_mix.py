import torch
import torch.nn as nn

import torch.distributions as D
from .distributions import Normal, MixtureSameFamily, TransformedDistribution
from .utils import clamp_preserve_gradients
from typing import List, Optional, Sequence, Tuple, Union

SLICE_IDX_T = Union[int, slice, type(Ellipsis)]
class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    We model it in the following way (see Appendix D.2 in the paper):

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
    """
    def __init__(
        self,
        locs: Optional[torch.Tensor] = None,
        log_scales: Optional[torch.Tensor] = None,
        log_weights: Optional[torch.Tensor] = None,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        direct_args: Optional[Tuple[D.Distribution, List[D.Transform]]] = None,
    ):
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        if direct_args is None:
            mixture_dist = D.Categorical(logits=log_weights)
            component_dist = Normal(loc=locs, scale=log_scales.exp())
            GMM = MixtureSameFamily(mixture_dist, component_dist)
            if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
                transforms = []
            else:
                transforms = [D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)]
            transforms.append(D.ExpTransform())

            direct_args = (GMM, transforms)

        super().__init__(*direct_args)

    @property
    def mean(self) -> torch.Tensor:
        """
        Compute the expected value of the distribution.

        See https://github.com/shchur/ifl-tpp/issues/3#issuecomment-623720667

        Returns:
            mean: Expected value, shape (batch_size, seq_len)
        """
        a = self.std_log_inter_time
        b = self.mean_log_inter_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights + a * loc + b + 0.5 * a**2 * variance).logsumexp(-1).exp()

    def __getitem__(self, index: Union[SLICE_IDX_T, Sequence[SLICE_IDX_T]]):
        if not isinstance(index, tuple): index = (index,)
        # We need to ensure the last axis is left unchanged, as that is the mixture aspect.
        index = index + (slice(None),)

        transforms = self.transforms
        base_dist = self.base_dist

        mixture_dist = base_dist.mixture_distribution
        mixture_logits = mixture_dist.logits

        curr_shape = mixture_logits.shape
        assert len(index) <= len(curr_shape), "Can't slice out the mixture axis!"

        sliced_logits = mixture_logits[index]
        sliced_mixture_dist = D.Categorical(logits=sliced_logits)

        component_dist = base_dist.component_distribution
        component_loc = component_dist.loc
        component_scale = component_dist.scale
        sliced_component_dist = Normal(
            loc = component_loc[index],
            scale = component_scale[index],
        )

        sliced_GMM = MixtureSameFamily(sliced_mixture_dist, sliced_component_dist)
        return type(self)(
            mean_log_inter_time=self.mean_log_inter_time, std_log_inter_time=self.std_log_inter_time,
            direct_args=(sliced_GMM, transforms)
        )
