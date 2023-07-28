import numpy as np
from tqdm import tqdm


distances_mean = []
distances_interp = []
distances_baseline = []
for trial in tqdm(range(100)):
    a = np.random.rand(2, 2)
    sigma = np.dot(a.T, a)
    x = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=100)
    d = np.sum(np.square(np.expand_dims(x, axis=1) - x), axis=-1)
    d = np.round(d, 2)
    indices = [(i, j) for i in range(d.shape[0]) for j in range(d.shape[1])]
    np.random.shuffle(indices)
    percentage = 0.3
    num_entries_to_replace = int(len(indices) * percentage)
    dd = np.copy(d)

    for i in range(num_entries_to_replace):
        x, y = indices[i]
        dd[x, y] = 0

    mean = np.mean(dd)
    dd_mean = np.copy(dd)
    dd_baseline = np.copy(dd)

    for i in range(num_entries_to_replace):
        x, y = indices[i]
        dd_mean[x, y] = mean

    for i in range(num_entries_to_replace):
        x, y = indices[i]
        if x == y:
            continue

        v = dd[x, y]

        #row = np.delete(dd[:,x], y, axis=0)
        row = dd[:,x]
        mask = row < np.quantile(row, q=0.4)
        weights = row * mask
        weights = weights / (np.sum(weights) + 1e-8)

        #v_interp = np.sum(np.delete(dd[y,:], x, axis=0) * weights)
        v_interp = np.sum(dd[y,:] * weights)
        dd[x, y] = v_interp

    distances_mean.append(np.sum(np.fabs(d - dd_mean)))
    distances_interp.append(np.sum(np.fabs(d - dd)))
    distances_baseline.append(np.sum(np.fabs(d - dd_baseline)))

print(f'baseline: {np.mean(distances_baseline)} ± {np.std(distances_baseline)}')
print(f'mean: {np.mean(distances_mean)} ± {np.std(distances_mean)}')
print(f'interp: {np.mean(distances_interp)} ± {np.std(distances_interp)}')


#print('d orig')
#print(d)
#
#np.random.shuffle(indices)
#for i in range(num_entries_to_replace):
#    x, y = indices[i]
#    d[x, y] = 0
#
#
#w = d / np.sum(d, axis=1, keepdims=True)
#
#iters = 0
#dd = np.copy(d)
#while iters < 4:
#    dd = np.dot(w.T, dd)
#    #w = dd / np.sum(dd, axis=1, keepdims=True)
#    iters += 1
#
#dd = np.round(dd, 2)
#print('d')
#print(d)
#print('dd')
#print(dd)

