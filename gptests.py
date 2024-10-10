import torch
import gpytorch
import numpy as np
import tqdm
from matplotlib import pyplot as plt


def heteroskedastic_data(x):
    return torch.hstack(
        [
            torch.normal(torch.sin(2 * np.pi * x), torch.exp(torch.cos(np.pi * x))),
            torch.normal(-torch.cos(2 * np.pi * x), torch.exp(torch.sin(np.pi * x))),
            torch.normal(
                x, torch.heaviside(torch.sin(4 * np.pi * x), torch.zeros_like(x))
            ),
            torch.normal(
                torch.heaviside(torch.sin(np.pi * x), torch.zeros_like(x)), 0.2 * x
            ),
        ]
    )


train_x = torch.linspace(0, 2.5, 1000).unsqueeze(-1)
train_y = heteroskedastic_data(train_x)
test_x = torch.linspace(0, 2.5, 250).unsqueeze(-1)
test_y = heteroskedastic_data(test_x)

batch_size = 256
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)


class NGD(torch.optim.Optimizer):

    def __init__(self, params, num_data, lr):
        self.num_data = num_data
        super().__init__(params, defaults=dict(lr=lr))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.add_(param.grad, alpha=(-group["lr"] * self.num_data))


class VariationalELBO(gpytorch.module.Module):

    def __init__(self, likelihood, model, num_data):
        super().__init__()
        self.likelihood = likelihood
        self.model = model
        self.num_data = num_data

    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        return self.likelihood.expected_log_prob(
            target, variational_dist_f, **kwargs
        ).sum()

    def forward(self, approximate_dist_f, target, **kwargs):
        num_batch = approximate_dist_f.event_shape[0]
        log_likehood = self._log_likelihood_term(approximate_dist_f, target).div(
            num_batch
        )
        kl_divergence = self.model.variational_strategy.kl_divergence().div(
            self.num_data
        )

        return log_likehood - kl_divergence


class ChainedGaussianLikelihood(gpytorch.module.Module):

    def __init__(self):
        super().__init__()

    def _draw_likelihood_samples(self, function_dist):
        sample_shape = torch.Size([100])
        if self.training:
            num_event_dims = len(function_dist.event_shape)
            function_dist = gpytorch.distributions.base_distributions.Normal(
                function_dist.mean, function_dist.variance.sqrt()
            )
            function_dist = gpytorch.distributions.base_distributions.Independent(
                function_dist, num_event_dims - 1
            )
        function_samples = function_dist.rsample(sample_shape)
        return self.forward(function_samples)

    def expected_log_prob(self, observations, function_dist):
        likelihood_samples = self._draw_likelihood_samples(function_dist)
        return likelihood_samples.log_prob(observations).mean(dim=0)

    def forward(self, function_samples):
        mean = function_samples[..., ::2]
        link = torch.nn.Softplus()
        scale = link(function_samples[..., 1::2]) + 1e-3
        return gpytorch.distributions.base_distributions.Normal(mean, scale)

    def log_marginal(self, observations, function_dist):
        raise NotImplementedError

    def marginal(self, function_dist):
        return self._draw_likelihood_samples(function_dist)

    def __call__(self, input_):
        # Conditional
        if torch.is_tensor(input_):
            return super().__call__(input_)
        # Maarginal
        elif isinstance(input_, gpytorch.distributions.MultivariateNormal):
            return self.marginal(input_)
        # Error
        else:
            raise RuntimeError(
                "Likelihoods expects a MultivariateNormal input to make marginal predictions, or a"
                " torch.Tensor for conditional predictions. Got a {}".format(
                    input_.__class__.__name__
                )
            )


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
                jitter_val=1e-3,
            ),
            num_tasks=output_dims,
            num_latents=num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.RBFKernel(
            batch_shape=batch_shape
        ) + gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=batch_shape)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


output_dims = 4
input_dims = 1
num_latents = 10
num_inducing = 20
model = MultitaskGPModel(
    num_latents=num_latents,
    output_dims=2 * output_dims,
    input_dims=input_dims,
    num_inducing=num_inducing,
)

likelihood = ChainedGaussianLikelihood()
model.train()
likelihood.train()

variational_ngd_optimizer = NGD(
    model.variational_parameters(), num_data=train_y.size(0), lr=0.1
)

hyperparameter_optimizer = torch.optim.Adam(model.hyperparameters(), lr=0.01)
# "Loss" for GPs - We are using the Variational ELBO
mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))

num_epochs = 150

epoch_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epoch_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        variational_ngd_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        minibatch_iter.set_postfix(loss=f"{loss.item():.2e}")
        loss.backward()
        variational_ngd_optimizer.step()
        hyperparameter_optimizer.step()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean.mean(dim=0)
    median = observed_pred.icdf(torch.Tensor([0.5])).mean(dim=0)
    lower = observed_pred.icdf(torch.Tensor([0.025])).mean(dim=0)
    upper = observed_pred.icdf(torch.Tensor([0.975])).mean(dim=0)

# Determine grid size for subplots
cols = int(np.ceil(np.sqrt(output_dims)))
rows = int(np.ceil(output_dims / cols))

fig, axs = plt.subplots(
    rows, cols, figsize=(4 * cols, 4 * rows)
)  # Adjust subplot size to maintain squareness

# Flatten the axes array for easy iteration if rows*cols > 1
axs = axs.ravel()

for task in range(output_dims):
    ax = axs[task]  # Get the appropriate subplot

    # Plot test and training data
    ax.plot(test_x.squeeze(-1).numpy(), test_y[:, task].numpy(), "r*")
    ax.plot(train_x.squeeze(-1).numpy(), train_y[:, task].numpy(), "k*")

    # Plot predictive means as blue line
    ax.plot(test_x.squeeze(-1).numpy(), mean[:, task].numpy(), "b")
    ax.plot(test_x.squeeze(-1).numpy(), median[:, task].numpy(), "g")

    # Shade between the lower and upper confidence bounds
    ax.fill_between(
        test_x.squeeze(-1).numpy(),
        lower[:, task].numpy(),
        upper[:, task].numpy(),
        alpha=0.5,
    )

    ax.set_xmargin(0)
    ax.set_title(f"Output {task + 1}")
axs[0].legend(
    ["Test Data", "Training Data", "Mean", "Median", "Confidence"]
)  # Add the legend to each subplot

# Remove empty subplots if output_dims < rows * cols
for idx in range(output_dims, rows * cols):
    fig.delaxes(axs[idx])

fig.tight_layout()
plt.show()
