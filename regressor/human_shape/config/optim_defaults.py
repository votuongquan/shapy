from typing import Tuple, List, Optional
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SGD:
    momentum: float = 0.9
    nesterov: bool = True


@dataclass
class ADAM:
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    amsgrad: bool = False


@dataclass
class RMSProp:
    alpha: float = 0.99


@dataclass
class Scheduler:
    type: str = 'none'
    gamma: float = 0.1
    milestones: Optional[Tuple[int]] = field(default_factory=tuple)
    step_size: int = 1000
    warmup_factor: float = 1.0e-1 / 3
    warmup_iters: int = 500
    warmup_method: str = 'linear'


@dataclass
class OptimConfig:
    type: str = 'adam'
    lr: float = 1e-4
    gtol: float = 1e-8
    ftol: float = -1.0
    maxiters: int = 100
    num_epochs: int = 300
    step: int = 30000
    weight_decay: float = 0.0
    weight_decay_bias: float = 0.0
    bias_lr_factor: float = 1.0

    sgd: SGD = field(default_factory=SGD)
    adam: ADAM = field(default_factory=ADAM)
    rmsprop: RMSProp = field(default_factory=RMSProp)

    scheduler: Scheduler = field(default_factory=Scheduler)
    #  discriminator: Optional[OptimConfig] = None


@dataclass
class OptimConfigWithDisc(OptimConfig):
    discriminator: OptimConfig = field(default_factory=lambda: OptimConfig())


conf = OmegaConf.structured(OptimConfigWithDisc)