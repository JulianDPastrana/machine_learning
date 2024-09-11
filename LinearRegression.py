import torch
from torch import nn
from GPNet.utilities import Module, DataModule, Trainer, add_to_class


# Create Synthetic DataModule
class SyntheticRegressionData(DataModule):
    """Synthetic data for linear regression."""

    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise


@add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)


# Create the model
class LinearRegression(Module):
    """The linear regression model implemented from API."""

    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)


@add_to_class(LinearRegression)
def forward(self, X):
    return self.net(X)


# Defining the Loss Function
@add_to_class(LinearRegression)
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)


# Defining the OptimizationAlgorithm
@add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)


# Trainig the model
model = LinearRegression(lr=0.03)
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = Trainer(max_epochs=3)
trainer.fit(model, data)


@add_to_class(LinearRegression)
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)


w, b = model.get_w_b()

print(f"error in estimating w: {data.w - w.reshape(data.w.shape)}")
print(f"error in estimating b: {data.b - b}")
