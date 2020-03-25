import numpy as np
import itertools
from numpy import linalg
import matplotlib as mpl

def generate_regression_data(num_samples):
	"""
	Generates baseline regression data.

	"""
	x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, num_samples))).T
	r_data = np.float32(np.random.normal(size=(num_samples,1)))
	y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)

	# switch x and y
	temp_data = x_data
	x_data = y_data
	y_data = temp_data

	return x_data, y_data

def plot_results(ax, X, means, covariances, index, title):
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        covar = torch.mm(covar, covar.permute(1, 0)).detach().numpy()
        v, w = linalg.eigh(3*covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color="#348ABD")
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, c="#E24A33")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.title(title)