import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, TrilNaturalVariationalDistribution
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 512)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * np.pi)) + torch.randn(train_x.size()) * np.sqrt(0.04)

test_x = torch.linspace(0, 1.5, 50)
test_y = torch.sin(2 * np.pi * test_x)

batch_size = 32
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = TrilNaturalVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SGD(Optimizer):
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                param.data -= group["lr"] * param.grad.data
            

    def zero_grad(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()


num_inducing_points = 15
inducing_points = torch.linspace(0, 1, num_inducing_points)
model = GPModel(inducing_points=inducing_points)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Find optimal model hyperparameters
model.train()
likelihood.train()
# Use custom optimizer
# optimizer = SGD(
#     params=[{"params": model.parameters()}, {"params": likelihood.parameters()}],
#     lr=0.05
# )

variational_ngd_optimizer = gpytorch.optim.NGD(
    model.variational_parameters(),
    num_data=train_y.size(0),
    lr=0.1
)

hyperparameter_optimizer = torch.optim.Adam(
    [
        {"params": model.hyperparameters()},
        {"params": likelihood.parameters()}
    ],
    lr=0.01
)
# "Loss" for GPs - We are using the Variational ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

num_epochs = 150
epoch_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epoch_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        variational_ngd_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()
        output = likelihood(model(x_batch))
        loss = -mll(output, y_batch)
        minibatch_iter.set_postfix(loss=loss.item())
        loss.backward()
        variational_ngd_optimizer.step()
        hyperparameter_optimizer.step()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), "k*")
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(["Observed Data", "Mean", "Confidence"])
    ax.set_xmargin(0)
    plt.show()
