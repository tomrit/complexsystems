import numpy as np
import matplotlib.pylab as plt


def calculate_dimension(data, epsilon, q=1):
    # define bins
    bin_edges = np.arange(-1.5, 1.5, epsilon)
    # calculate 2D histogram
    histogram, x_edges, y_edges = np.histogram2d(data[:, 0], data[:, 1], bin_edges)
    # normalize
    probability = histogram / data.size
    # exclude equal zero elements
    probability_positive = probability[0 < probability]
    # Renyi information
    if q == 1:
        I = -np.sum(probability_positive * np.log(probability_positive))
    else:
        I = 1 / (1 - q) * np.log(np.sum(probability_positive ** q))

    return I


def iterate_epsilon(data, q, epsilons):
    Is = np.zeros((epsilons.size, 1))

    for idx, epsilon in enumerate(epsilons):
        Is[idx] = calculate_dimension(data, epsilon, q)

    return Is


def iterate_data_size(data, q, epsilons, ns):
    Is = np.zeros((ns.size, epsilons.size, 1))
    for idx, n in enumerate(ns):
        Is[idx] = iterate_epsilon(data[1:n, :], q, epsilons)

    return Is


def visualize_data(data_points):
    fig1, ax1 = plt.subplots()
    ax1.plot(data_points[:, 0], data_points[:, 1], ',')


def visualize_iterated_i(data_points):
    epsilons = np.logspace(-3.5, 0, 40)
    ns = np.linspace(1e4, 1e5, 7).astype(int)
    qs = np.array([0, 1, 2])

    fig2, ax_array = plt.subplots(1, qs.size, sharey=True, figsize=(12, 8))
    ax_array_flat = ax_array.reshape(-1)
    for idx, q in enumerate(qs):
        Is = iterate_data_size(data_points, q, epsilons, ns)
        ax2 = ax_array_flat[idx]
        for idx2, I in enumerate(Is):
            n = ns[idx2]
            ax2.plot(np.log10(1 / epsilons), I, 'o-', label=str(n))
        ax2.legend(loc='best')
        ax2.set_xlabel('$log(1/\epsilon)$')
        ax2.set_ylabel('$I(\epsilon)$')
        ax2.set_title('$q = {}$'.format(q))

    ns_string = "n = " + ", ".join("{}".format(n) for n in ns)
    qs_string = "q = " + ", ".join("{}".format(q) for q in qs)
    fig2.savefig('./graphics/information_dimension {} {}.png'.format(ns_string, qs_string), dpi=300, transparent=True,
                 bbox_inches='tight')

if __name__ == '__main__':
    data_points = np.loadtxt("data1.txt")
    # visualize_data(data_points)
    visualize_iterated_i(data_points)
    plt.show()
