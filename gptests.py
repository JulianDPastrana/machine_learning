import sys
import torch
import gpytorch
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader


# Function to generate heteroskedastic noise (noise increasing with x)
def heteroskedastic_noise(x, base_std=0.1, scale=0.5):
    return torch.randn(x.size()) * torch.cos(6*np.pi*x)



# Training data: 512 points in [0,1] inclusive, regularly spaced
train_x = torch.linspace(0, 1, 10*1024).unsqueeze(-1)

# Generating the training labels with heteroskedastic noise
train_y = torch.hstack(
    [
        torch.sin(train_x * (2 * np.pi))
        + heteroskedastic_noise(train_x, base_std=0.1, scale=0.5),
        -torch.cos(train_x * (2 * np.pi))
        + heteroskedastic_noise(train_x, base_std=0.1, scale=0.5),
        torch.sin(train_x * (2 * np.pi))
        + 2 * torch.cos(train_x * (2 * np.pi))
        + heteroskedastic_noise(train_x, base_std=0.1, scale=0.5),
        train_x + heteroskedastic_noise(train_x, base_std=0.1, scale=0.5),
    ]
)

# Test data: 50 points in [0, 1.5] regularly spaced
test_x = torch.linspace(0, 1.25, 50).unsqueeze(-1)

# Generating the test labels with heteroskedastic noise
test_y = torch.hstack(
    [
        torch.sin(test_x * (2 * np.pi))
        + heteroskedastic_noise(test_x, base_std=0.1, scale=0.5),
        -torch.cos(test_x * (2 * np.pi))
        + heteroskedastic_noise(test_x, base_std=0.1, scale=0.5),
        torch.sin(test_x * (2 * np.pi))
        + 2 * torch.cos(test_x * (2 * np.pi))
        + heteroskedastic_noise(test_x, base_std=0.1, scale=0.5),
        test_x + heteroskedastic_noise(test_x, base_std=0.1, scale=0.5),
    ]
)

batch_size = 1024
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class ChainedGaussianLikelihood(gpytorch.likelihoods.Likelihood):
    def __init__(self, output_dims):
        super().__init__()
        self.output_dims = output_dims
        # self.num_params = 2 * output_dims

    def forward(self, function_samples):
        mean = function_samples[..., ::2]
        variance = torch.exp(function_samples[..., 1::2])
        return torch.distributions.normal.Normal(loc=mean, scale=variance)

    # def expected_log_prob(observations, function_dist):
    #     log_prob_lambda = lambda function_samples: self.forward(
    #         function_samples
    #     ).log_prob(observations)
    #     return self.quadrature(log_prob_lambda, function_dist)


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_latents, output_dims, input_dims, num_inducing):
        batch_shape = torch.Size([num_latents])
        inducing_points = torch.randn(num_latents, num_inducing, input_dims)
        variational_distribution = (
            gpytorch.variational.TrilNaturalVariationalDistribution(
                num_inducing_points=num_inducing, batch_shape=batch_shape
            )
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=output_dims,
            num_latents=num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


output_dims = 4
input_dims = 1
num_latents = 8
num_inducing = 32
model = MultitaskGPModel(
    num_latents=num_latents,
    output_dims=2*output_dims,
    input_dims=input_dims,
    num_inducing=num_inducing,
)

# initialize likelihood and model
likelihood = ChainedGaussianLikelihood(output_dims=output_dims)
# print(likelihood(model(train_x)).expected_log_prob(train_y))

# sys.exit()
# Find optimal model hyperparameters
model.train()
likelihood.train()

variational_ngd_optimizer = gpytorch.optim.NGD(
    model.variational_parameters(), num_data=train_y.size(0), lr=0.1
)

hyperparameter_optimizer = torch.optim.Adam(
    model.hyperparameters(), lr=0.01
)
# "Loss" for GPs - We are using the Variational ELBO
mll = gpytorch.mlls.DeepApproximateMLL(
    gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
)

num_epochs = 50
epoch_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epoch_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        variational_ngd_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()
        output = model(x_batch)
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
    print(observed_pred)
    mean = observed_pred.mean.mean(0)
    variance = observed_pred.variance.mean(0)
    lower = mean - 1.96 * variance.sqrt()
    upper = mean + 1.96 * variance.sqrt()

cols = int(np.ceil(np.sqrt(output_dims)))
rows = int(np.ceil(output_dims / cols))

fig, axs = plt.subplots(
    rows, cols, figsize=(4 * cols, 4 * rows)
)  # Adjust subplot size to maintain squareness

# Flatten the axes array in case rows*cols > 1 for easy iteration
axs = axs.ravel()

for task in range(output_dims):
    ax = axs[task]  # Get the appropriate subplot

    # Initialize plot
    # Plot training data as black stars
    ax.plot(test_x.squeeze(-1).numpy(), test_y[:, task].numpy(), "k*")

    # Plot predictive means as blue line
    ax.plot(test_x.squeeze(-1).numpy(), mean[:, task].numpy(), "b")

    # Shade between the lower and upper confidence bounds
    ax.fill_between(
        test_x.squeeze(-1).numpy(),
        lower[:, task].numpy(),
        upper[:, task].numpy(),
        alpha=0.5,
    )

    ax.legend(["Observed Data", "Mean", "Confidence"])
    ax.set_xmargin(0)
    ax.set_title(f"Output {task + 1}")

# Remove empty subplots if output_dims < rows * cols
for idx in range(output_dims, rows * cols):
    fig.delaxes(axs[idx])

fig.tight_layout()
plt.show()
