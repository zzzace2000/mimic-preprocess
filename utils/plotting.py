import matplotlib.pyplot as plt
import numpy as np


def sparse_matrix_to_list(t, y, ind_kt, ind_kf, num_features=None):
    if num_features is None:
        num_features = int(max(ind_kf) + 1)
    num_measurements = len(y)
    data = [[[], []] for _ in range(num_features)]
    for i in range(num_measurements):
        data[int(ind_kf[i])][1].append(y[i])
        data[int(ind_kf[i])][0].append(t[int(ind_kt[i])])
    return data


def plot_measurement(t, y, ind_kt, ind_kf, x=None, y_upper=None, y_lower=None, title=None, feature_names=None):
    num_features = int(max(ind_kf) + 1)
    if feature_names is not None:
        num_features = len(feature_names)

    print('num_features:', num_features)

    t_min = min(t)
    t_max = max(t)

    f, ax = plt.subplots(num_features, figsize=(15, 60), sharex=True)

    if title is not None:
        plt.title(title)

    # Collect data
    data = sparse_matrix_to_list(t, y, ind_kt, ind_kf, num_features)

    if y_upper is not None and y_lower is not None:
        data_upper = sparse_matrix_to_list(t, y_upper, ind_kt, ind_kf, num_features)
        data_lower = sparse_matrix_to_list(t, y_lower, ind_kt, ind_kf, num_features)

    # Plotting
    for m in range(num_features):
        ax[m].scatter(data[m][0], data[m][1])
        ax[m].axvline(t_min, color='r', linestyle='--')
        ax[m].axvline(t_max, color='r', linestyle='--')
        if x is not None:
            x_min = min(x)
            x_max = max(x)
            ax[m].axvline(x_min, color='g', linestyle='--')
            ax[m].axvline(x_max, color='g', linestyle='--')

        if y_upper is not None and y_lower is not None:
            ax[m].fill_between(data_upper[m][0], data_lower[m][1], data_upper[m][1], alpha=0.1)

        if feature_names is not None:
            ax[m].set_xlabel(feature_names[m])
    plt.show()


def plot_measurement_dense(mc_obs, t=None, y=None, ind_kt=None, ind_kf=None, x=None, x_len=None, feature_names=None):
    obs_mean = np.mean(mc_obs, axis=0)[:int(x_len),:]
    obs_std = np.std(mc_obs, axis=0)[:int(x_len),:]
    obs_upper = obs_mean + obs_std
    obs_lower = obs_mean - obs_std
    num_features = int(max(ind_kf) + 1)

    if feature_names is not None:
        assert len(feature_names) >= num_features, \
            'wierd! feature name only have %d but with this %d much features.' \
            % (len(feature_names), num_features)

    x_min = min(x)
    x_max = max(x)
    t_min = t[ind_kt[0]]
    t_max = max(t)

    data = sparse_matrix_to_list(t, y, ind_kt, ind_kf, num_features=num_features)

    f, ax = plt.subplots(num_features, figsize=(15, 30), sharex=True)

    for m in range(num_features):
        ax[m].scatter(x[:int(x_len)], obs_mean[:,m])
        ax[m].fill_between(x[:int(x_len)], obs_lower[:, m], obs_upper[:, m], alpha=0.1)
        ax[m].scatter(data[m][0], data[m][1])
        ax[m].axvline(t_min, color='r', linestyle='--')
        ax[m].axvline(t_max, color='r', linestyle='--')
        ax[m].axvline(x_min, color='g', linestyle='--')
        ax[m].axvline(x_max, color='g', linestyle='--')
        if feature_names is not None:
            ax[m].set_xlabel(feature_names[m])

    plt.show()

def _plot_element(**kwargs):
    if 'title' in kwargs: plt.title(kwargs['title'], fontsize=20)
    if 'lower' in kwargs is not None and kwargs['upper'] is not None:
        plt.fill_between(kwargs['x'], kwargs['lower'], kwargs['upper'] , alpha=0.2)
    if 'xlabel' in kwargs: plt.xlabel(kwargs['xlabel'], fontsize=20)
    if 'ylabel' in kwargs: plt.ylabel(kwargs['ylabel'], fontsize=20)
    if 'xlim' in kwargs: plt.xlim(kwargs['xlim'])
    if 'ylim' in kwargs: plt.ylim(kwargs['ylim'])


def plot_line_simple(x, y, **kwargs):
    plt.figure()
    plt.plot(x, y)
    plt.grid(True)
    _plot_element(**dict({'x': x}, **kwargs))
    plt.show()


def plot_scatter_simple(x, y, **kwargs):
    plt.figure()
    plt.scatter(x, y)
    plt.grid(True)
    _plot_element(**kwargs)
    plt.show()


def plot_hist_simple(data, bins=None, range=None, **kwargs):
    plt.figure()
    plt.hist(data, bins=bins, range=range)
    plt.grid(True)
    _plot_element(**kwargs)
    plt.show()


def plot_box_simple(data, **kwargs):
    """
    >>> import numpy as np
    >>> x = np.arange(20).reshape(10, 2)
    >>> plot_box_simple(x) # plot two box plot side by side
    """
    plt.figure()
    plt.boxplot(data, autorange=True)
    plt.grid(True)
    _plot_element(**kwargs)
    plt.show()


def plot_hist(data, components=None, names=None):
    if components is not None:
        assert max(components) < data.shape[1]
        f, ax = plt.subplots(len(components), sharey=False, sharex=False, tight_layout=True)
        i = 0
        for m in components:
            ax[i].hist(data[:, m])
            ax[i].grid(True)
            if names is not None:
                ax[i].set_xlabel(names[i])
            i +=1
        plt.show()


def plot_scatter(data, ax1=0, ax2=1, group_ax=None, split=False):
    if not split:
        plt.figure()
        sc = plt.scatter(x=data[:,ax1], y=data[:,ax2], c=data[:,group_ax], cmap=plt.jet(), alpha=0.5, s=1)
        plt.colorbar(sc)
    else:
        num_groups = int(1 + max(data[:, group_ax]))
        f, ax = plt.subplots(num_groups, figsize=(15, 30), sharey=True, sharex=True, tight_layout=True)
        for m in range(num_groups):
            idx = np.arange(data.shape[0])[data[:, group_ax] == m]
            ax[m].scatter(x=data[idx, ax1], y=data[idx, ax2], alpha = 0.6)
            ax[m].grid(True)
    plt.show()

if __name__ == '__main__':
    pass