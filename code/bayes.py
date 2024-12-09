import numpy as np

# generate means
blu_means = np.random.multivariate_normal(np.array([1, 0]), np.eye(2), size=10)
org_means = np.random.multivariate_normal(np.array([0, 1]), np.eye(2), size=10)


# generate samples
def generate(mus, n_samples=100):
    samples = []
    for _ in range(n_samples):
        # sample mean at random
        idx = np.random.choice(mus.shape[0])
        # sample from Gaussian
        samples.append(np.random.multivariate_normal(mus[idx], np.eye(2) / 5.0))

    return samples


def pdf(x, mu, sigma):
    # only care about unnormalized PDF since the covariances are identical
    return np.exp(-np.dot(x - mu, x - mu) / (2 * sigma))


def pdf_blue(x, y):
    sigma = 1.0 / 5.0
    val = 0.0
    for mu in blu_means:
        val += pdf((x, y), mu, sigma)
    return 1.0 / 10.0 * val


def pdf_orange(x, y):
    sigma = 1.0 / 5.0
    val = 0.0
    for mu in org_means:
        val += pdf((x, y), mu, sigma)
    return 1.0 / 10.0 * val


# sample from the distributions
blue_samples = np.array(generate(blu_means, 100))
orange_samples = np.array(generate(org_means, 100))

# create the contour maps
vpdf_blue = np.vectorize(pdf_blue)
vpdf_orange = np.vectorize(pdf_orange)

min_x, max_x, min_y, max_y = -4, 4, -4, 4
xs, ys = np.meshgrid(np.linspace(min_x, max_x, 500), np.linspace(min_y, max_y, 500))
grid_blue = vpdf_blue(xs, ys)
grid_orange = vpdf_orange(xs, ys)

# Bayes boundary
bdry = np.where(grid_blue > grid_orange, 0, 2)
bdry[np.abs(grid_blue - grid_orange) < 1e-6] = 1
