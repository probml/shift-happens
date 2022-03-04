import flax.linen as nn

class MLPDataV1(nn.Module):
    num_outputs: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(800)(x))
        x = nn.relu(nn.Dense(500)(x))
        x = nn.Dense(self.num_outputs)(x)
        x = nn.log_softmax(x)
        return x


class MLPWeightsV1(nn.Module):
    num_outputs: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(200)(x))
        x = nn.Dense(self.num_outputs)(x)
        return x

