import jax
from functools import partial

from agents.mlp import MLP
from agents.lenet import LeNet5
from agents.resnet import ResNet, BottleneckResNetBlock, ResNetBlock

ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3],
                    block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3],
                    block_cls=BottleneckResNetBlock)
LeNet = partial(LeNet5, act=jax.nn.relu)

mlp = partial(MLP, layer_dims=[256, 256])

# Used for testing only.
_ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
