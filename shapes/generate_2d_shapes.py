import matplotlib.pyplot as plt
import numpy as np 


def generate_line(m, b, x_range, step_size, noise_std=0.05, seed=None):
    """Generate points around the line y = m*x + b.

    ``noise_std`` controls the thickness of the line. Set it to 0.0 for a
    perfectly defined line. ``seed`` makes the random noise reproducible.
    """
    point_count = int(x_range // step_size) + 1
    x = np.linspace(0, x_range, point_count)
    y = m * x + b
    points = np.column_stack((x, y))

    rng = np.random.default_rng(seed)
    points += rng.normal(0, noise_std, size=points.shape)
    return points

def generate_spiral(a, b, length, noise_std=0.05, seed=None):
    """Generate a logarithmic spiral described by r = a*exp(b*theta).

    ``a`` sets the initial radius, ``b`` controls how quickly the spiral
    expands, and ``length`` controls its number of half-turns. ``noise_std``
    gives the curve thickness; use 0.0 for a perfectly defined spiral.
    """
    theta = np.arange(0, length * np.pi, 0.1)
    radius = a * np.exp(b * theta)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    points = np.column_stack((x, y))
    rng = np.random.default_rng(seed)
    points += rng.normal(0, noise_std, size=points.shape)
    return points


def generate_single_gaussian(n_points, mean=(0, 0), std=1.0, seed=None):
    """Generate one cloud centered at ``mean`` with spread ``std``.

    Unlike the curve generators, the Gaussian's ``std`` is both its defining
    parameter and its natural data noise.
    """
    rng = np.random.default_rng(seed)
    mean = np.asarray(mean)
    return rng.normal(loc=mean, scale=std, size=(n_points, 2))


def generate_two_gaussian_clusters(
    n_points,
    centers=((-2, 0), (2, 0)),
    std=0.5,
    seed=None,
):
    """Generate two separate Gaussian clouds around the given ``centers``.

    ``std`` controls the spread within both clusters. The points are shuffled
    so their array order does not reveal which cluster they came from.
    """
    rng = np.random.default_rng(seed)
    first_count = n_points // 2
    second_count = n_points - first_count

    first_cluster = rng.normal(centers[0], std, size=(first_count, 2))
    second_cluster = rng.normal(centers[1], std, size=(second_count, 2))
    points = np.vstack((first_cluster, second_cluster))
    rng.shuffle(points)
    return points


def generate_circle(n_points, radius=1.0, noise_std=0.05, seed=None):
    """Generate evenly spaced points around a circle.

    ``radius`` sets its size and ``noise_std`` gives its edge thickness. Set
    ``noise_std`` to 0.0 for a perfectly defined circumference.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    points += rng.normal(0, noise_std, size=points.shape)
    return points


def generate_concentric_circles(
    n_points,
    radii=(1.0, 2.0),
    noise_std=0.05,
    seed=None,
):
    """Generate points distributed among multiple circles.

    ``radii`` defines the circle sizes. ``noise_std`` gives every ring a small
    thickness and can be set to 0.0 for perfectly defined rings.
    """
    rng = np.random.default_rng(seed)
    circle_indices = rng.integers(0, len(radii), size=n_points)
    theta = rng.uniform(0, 2 * np.pi, size=n_points)
    point_radii = np.asarray(radii)[circle_indices]

    points = np.column_stack(
        (point_radii * np.cos(theta), point_radii * np.sin(theta))
    )
    points += rng.normal(0, noise_std, size=points.shape)
    return points


def generate_two_moons(n_points, noise_std=0.05, seed=None):
    """Generate two interleaving half-circles.

    The moons form two distinct curved groups. ``noise_std`` controls their
    thickness and can be set to 0.0 for perfectly defined arcs.
    """
    rng = np.random.default_rng(seed)
    first_count = n_points // 2
    second_count = n_points - first_count

    first_theta = rng.uniform(0, np.pi, size=first_count)
    first_moon = np.column_stack((np.cos(first_theta), np.sin(first_theta)))

    second_theta = rng.uniform(0, np.pi, size=second_count)
    second_moon = np.column_stack(
        (1 - np.cos(second_theta), 0.5 - np.sin(second_theta))
    )

    points = np.vstack((first_moon, second_moon))
    points += rng.normal(0, noise_std, size=points.shape)
    rng.shuffle(points)
    return points


def plot_figure(points): 
    fig, ax = plt.subplots() 
    x = points[:,0]
    y = points[:,1]
    ax.plot(x,y, 'o')
    ax.set_title("Simple Waveform Visualization", fontsize=14)

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    #ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def main():
    """Generate the golden spiral
    phi = (1 + np.sqrt(5)) /2
    a = 1e-42
    b = 2 * np.log(phi) / np.pi
    sp_points = generate_spiral(a,b,100)
    print_figure(sp_points)
    """
    points = generate_circle(1000, noise_std=0.01)
    plot_figure(points)


if __name__ == "__main__": 
    main()
