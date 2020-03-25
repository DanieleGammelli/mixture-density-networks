import torch
from torch import nn
import pyro
from pyro.infer import config_enumerate
import pyro.distributions as dist

class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, K):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = K
        self.tril_indices = torch.tril_indices(row=output_dim, col=output_dim, offset=-1)
        # initialize linear transformations
        self.lin_input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_mix_components = nn.Linear(hidden_dim, K)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, output_dim*K)
        self.lin_hidden_to_offdiag = nn.Linear(hidden_dim, K)
        self.lin_hidden_to_sigma = nn.Linear(hidden_dim, output_dim*K)
        
        # initialize non-linearities
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)
    
    def forward(self, x):
        h = self.relu(self.lin_input_to_hidden(x))
        h = self.bn1(self.relu(self.lin_hidden_to_hidden(h)))
        h = self.dropout1(h)
        h = self.bn2(self.relu(self.lin_hidden_to_hidden(h)))
        h = self.dropout2(h)
        pi = self.softmax(self.lin_hidden_to_mix_components(h))
        loc = self.lin_hidden_to_loc(h).view(-1, self.K, self.output_dim)
        sigma = self.softplus(self.lin_hidden_to_sigma(h)).view(-1, self.K, self.output_dim)
        offdiag = self.lin_hidden_to_offdiag(h).view(-1, self.K, 1)
        Sigma_tril = torch.zeros((x.shape[0], self.K, self.output_dim, self.output_dim), device=x.device)
        for i in range(self.K):
            Sigma_tril[:, i, self.tril_indices[0], self.tril_indices[1]] = offdiag[:, i, :]
            Sigma_tril[:, i] += torch.diag_embed(sigma[:, i, :])
        return pi, loc, Sigma_tril

class PyroMDN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=20, output_dim=1, K=3, use_cuda=False):
        super(PyroMDN, self).__init__()
        self.mdn = MDN(input_dim, hidden_dim, output_dim, K)
        
        self.K = K
        self.D = output_dim
        if use_cuda:
            self.cuda()
    
    @config_enumerate
    def model(self, X=None, y=None):
        N = X.shape[0]
        D = X.shape[1]
        pyro.module("MDN", self)
        pi, loc, Sigma_tril = self.mdn(X)
        locT = torch.transpose(loc, 0, 1)
        Sigma_trilT = torch.transpose(Sigma_tril, 0, 1)
        assert pi.shape == (N, self.K)
        assert locT.shape == (self.K, N, D)
        assert Sigma_trilT.shape == (self.K, N, D, D)
        with pyro.plate("data", N):
            assignment = pyro.sample("assignment", dist.Categorical(pi))
            if len(assignment.shape) == 1:
                _mu = torch.gather(locT, 0, assignment.view(1, -1, 1))[0]
                _scale_tril = torch.gather(Sigma_trilT, 0, assignment.view(1, -1, 1, 1))[0]
                sample = pyro.sample('obs', dist.MultivariateNormal(_mu, scale_tril=_scale_tril), obs=y)
            else: 
                _mu = locT[assignment][:,0]
                _scale_tril = Sigma_trilT[assignment][:,0]
                sample = pyro.sample('obs', dist.MultivariateNormal(_mu, scale_tril=_scale_tril), obs=y)
                
        return pi, loc, Sigma_tril, sample
    
    def guide(self, X=None, y=None):
        pass