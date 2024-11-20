import re
import numpy as np
from functools import reduce, lru_cache
from itertools import pairwise
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from python_tsp.exact import (
        solve_tsp_branch_and_bound,
        solve_tsp_dynamic_programming)
from time import perf_counter


class NodeArray:
    def __init__(self, nodedict, name='', parts=None):
        self.name = name
        self.parts = parts or {name: np.arange(len(nodedict))}
        self.names = np.array(list(nodedict))
        self.xyz = np.array(list(nodedict.values()))
        self._nodedict = nodedict
        self._i = {name: i for i, name in enumerate(nodedict)}

    def __getitem__(self, N):
        if isinstance(N, str):
            return self._i[N]
        try:
            return self.names[N]
        except IndexError as ie:
            if isinstance(N, int):
                raise ie
        return [self._i[n] for n in N]

    def __len__(self):
        return len(self._nodedict)

    def __or__(self, other):
        assert not (self._nodedict.keys() & other._nodedict.keys()), \
                "NodeArray keys overlap"
        parts = {}
        for part in self.parts:
            parts[part] = self.parts[part]
        for part in other.parts:
            parts[part] = parts.get(part, [])
            parts[part] = np.concatenate((parts[part],
                                          other.parts[part] + len(self)))
        return NodeArray(self._nodedict | other._nodedict, parts=parts)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text('{...}')
            return
        with p.group(1, '{', '}'):
            for name, xyz in self._nodedict.items():
                if self._i[name]:
                    p.text(',')
                    p.breakable()
                p.pretty(str(name))
                p.text(': ')
                p.pretty(xyz)

    def _ipython_key_completions_(self):
        return list(self.names)


def euclidean_distance(source, dest):
    if source.ndim == 1:
        source = source[None]
    d = source[:,None,:] - dest[None,...]
    d[d[...,1]>0, 1] = 0
    d = np.sum(d**2, axis=-1)**0.5
    return d


def disconnect(A, /, isle1, isle2=None, *, keep=None, debug=False):
    """
    Disconnects `isle1` from `isle2`. Connections to and from `keep` are kept.
    Assigning `None` to `isle2` is equivalent to assigning all nodes to them.
    Connections inside `isle1` are kept, but inside `isle2` won't be.
    This is relevant, when `isle1` overlaps with `isle2`, those will still be
    disconnected from `isle2`.

    `A` is the adjacency matrix.
    """
    # All outer connections to `isle1`
    I1 = np.zeros_like(A, dtype=bool)
    I1[isle1] = 1
    I1 = I1 ^ I1.T
    # All connections to `isle2`
    I2 = np.zeros_like(A, dtype=bool)
    I2[isle2] = I2[:,isle2] = 1
    # All connections to `keep`
    G = np.zeros_like(I1)
    if keep:
        G[keep] = G[:,keep] = 1
    # Removing connections
    if debug:
        plt.imshow(I1 & I2 & ~G)
        plt.show()
    A[I1 & I2 & ~G] = np.inf


def adjacency_matrix(nodes: NodeArray):
    A = euclidean_distance(nodes.xyz, nodes.xyz)

    # Realm of Light
    *realm_of_light, ls \
            = rl, *_ \
            = nodes[['Realm of Light', 'Tree of Light', 'Light\'s Secret']]
    disconnect(A, realm_of_light)

    # Molten Heights
    *molten_heights, en, sm \
            = ro, mh, *_ = nodes[['Rodoroc', 'Molten Heights',
                                  'Entrance to Molten Heights',
                                  'Sky-Molten Tunnel']]
    disconnect(A, molten_heights, keep=[en, sm])
    disconnect(A, [en], [ro, sm])

    # Sky Islands
    _, *sky_islands \
            = te, jt, kb, jd \
            = nodes[['Thesead-Eltom Tunnel', 'Jofash Tunnel',
                     'Kandon-Beda', 'Jofash Dock']]
    sky_islands.append(sm)
    disconnect(A, sky_islands, keep=[ro, mh, te, jd])

    # Jungle
    *jungle, tp, iv \
            = bb, tr, dt, gb, *_ \
            = nodes[['Basalt Basin', 'Troms', 'Dernel Tunnel',
                     'The Great Bridge', 'The Passage', 'Icy Vigil']]
    disconnect(A, jungle, keep=[tp,iv])
    disconnect(A, [tp], jungle, keep=[tr])
    disconnect(A, [iv], jungle, keep=[gb])
    disconnect(A, [bb], jungle, keep=[dt])

    # Gavel
    lg, ll, ww, *ocean = nodes[['Llevigar Gate', 'Llevigar',
                                'Weird Wilds', 'Selchar']]
    wynn = nodes[['Nesaak',  'Ragni-Detlas Tunnel',
                  'Detlas', 'Nemract', 'Tempo Town']] + [tp, iv]
    not_gavel = wynn + ocean + jungle
    disconnect(A, not_gavel, keep=[ll, ww, jd])
    disconnect(A, [ll], keep=[lg, ww, jd] + not_gavel)
    disconnect(A, [jd], keep=[ll, ww, jt] + not_gavel)
    return A


def fasttravel_matrix(nodes: NodeArray):
    F = np.full((len(nodes),)*2, np.inf)

    # Ragni-Detlas Tunnel
    ra, de = nodes[['Ragni-Detlas Tunnel', 'Detlas']]
    F[ra, de] = F[de, ra] = 0

    # V.S.S. Seaskipper
    nr, ns, se, ll, jd = nodes[['Nemract', 'Nesaak', 'Selchar',
                                'Llevigar', 'Jofash Dock']]
    vss = fasttravel['Seaskipper']
    F[nr, [ns,se,ll]] = vss
    F[ll, [ns,se,nr,jd]] = vss
    F[jd, [nr,se,ll]] = vss
    F[se, [nr,ll,ns,jd]] = vss
    F[ns, [nr,se,ll]] = vss

    # Mysterious Obelisk
    tt = nodes['Tempo Town']
    obelisk = fasttravel['Obelisk']
    F[nr, tt] = F[tt, nr] = obelisk

    # Nexus Hub | overwrites V.S.S. since faster
    hub = np.zeros_like(F, dtype=bool)
    hub[[nr, se, ll]] = 1
    F[hub & hub.T] = fasttravel['Nexus Hub']

    # Calo's Airship
    la = nodes['Letvus Airbase']
    F[de, la] = F[la, de] = fasttravel['Airship']

    # The Juggler's Tavern
    ci, at, th, ro = nodes[['Cinfras', 'Aldorei Town', 'Thesead', 'Rodoroc']]
    F[ci, [at, th, ro]] = fasttravel['Juggler']

    # Nexus Gate
    ls, rl = nodes[['Light\'s Secret', 'Realm of Light']]
    F[ls, rl] = fasttravel['Nexus of Light']
    F[rl, ls] = fasttravel['Light Portal']

    # Eltom-Thesead Tunnel
    el, th = nodes[['Eltom', 'Thesead-Eltom Tunnel']]
    F[el, th] = F[th, el] = 0

    # Dwarven Trading Tunnel
    tn = nodes['Thanos']
    F[tn, ro] = F[ro, tn] = 0

    # Colossus Tunnel
    ct, kb = nodes[['Colossus Tunnel', 'Kandon-Beda']]
    F[ct, kb] = F[kb, ct] = 0

    np.fill_diagonal(F, np.inf)
    return F


def floyd_warshall(A, return_predecessors=False):
    D = np.copy(A)
    if not return_predecessors:
        for k in range(D.shape[1]):
            D = np.minimum(D, D[:,k][:,None] + D[k,:])
        return D
    P = np.tile(np.arange(A.shape[0])[:,None], (1,A.shape[0]))
    P[np.isinf(A)] = -1
    for k in range(D.shape[1]):
        Dk = D[:,k][:,None] + D[k]
        change = np.nonzero(Dk < D)
        D[change] = Dk[change]
        P[change] = P[k,change[1]]
    return D, P


def follow_predecessors(P, src, dst):
    pre = P[src, dst].item()
    if pre == src:
        return [src, dst]
    path = follow_predecessors(P, src, pre)
    path.append(dst)
    return path


def expand_path(P, path):
    expanded = [path[0]]
    for i,j in pairwise(path):
        expanded.extend(follow_predecessors(P, i, j)[1:])
    return expanded


def path_length(A, path):
    return reduce(lambda l,e: l + A[e], pairwise(path), 0)


class TSP_Solver:
    """
    Source: https://github.com/fillipe-gsm/python-tsp
    """

    def __init__(self, distance_matrix: np.ndarray,
                 maxsize: Optional[int] = None):
        # adding a dummy node to convert path problem to cycle problem
        self._D = np.pad(distance_matrix, [(0,1), (0,1)],
                         mode='constant', constant_values=0)
        self.dist = lru_cache(maxsize)(self._dist)

    def _dist(self, ni: int, N: frozenset) -> float:
        if not N:
            return self._D[ni, 0]

        costs = [(nj, self._D[ni,nj] + self.dist(nj, N - {nj}))
                 for nj in N]
        nmin, min_cost = min(costs, key=lambda x: x[1])
        self._memo[(ni, N)] = nmin
        return min_cost

    def solve(self, *, for_path=False) -> tuple[list, float]:
        n = self._D.shape[0]
        if for_path:
            ni = n - 1  # start from the dummy node
            solution = []
        else:
            n -= 1  # remove the dummy node
            solution = [ni] = [0]

        N = frozenset(range(0, n)) - {ni}
        self._memo: dict[tuple, int] = {}

        # Step 1: get minimum distance
        best_distance = self.dist(ni, N).item()

        # Step 2: get path with the minimum distance
        while N:
            ni = self._memo[(ni, N)]
            solution.append(ni)
            N = N - {ni}

        return solution, best_distance


def prune(P, part):
    paths = set()
    edges = set()

    for (i,j), p in np.ndenumerate(P[part][:,part]):
        if p == -1:
            continue
        p = p.item()
        if p != i:
            paths.add((i,p))
        edges.add((p,j))

    while paths:
        fix_paths = set(paths)
        for i,j in fix_paths:
            p = P[i,j].item()
            if p == -1:
                continue
            if p != i:
                paths.add((i,p))
            edges.add((p,j))
        paths = paths - fix_paths

    return tuple(np.array(list(edges)).T)


def plot(nodes, A, F=None):
    if F is None:
        F = np.zeros_like(A, dtype=bool)
    else:
        F = ~np.isinf(F)
    A = ~np.isinf(A)
    fast = np.stack(np.nonzero(A & F)).T
    path = np.stack(np.nonzero(A & ~F)).T
    fast = nodes.xyz[:,[0,2]][fast]
    path = nodes.xyz[:,[0,2]][path]

    fc = mc.LineCollection(fast, linestyle=(0, (1, 5)))
    pc = mc.LineCollection(path)
    fig, ax = plt.subplots()

    plt.scatter(*nodes.xyz[:,[0,2]].T, c='y', zorder=10)
    ax.add_collection(fc)
    ax.add_collection(pc)

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show(block=False)


def plot_path(nodes, path, F=None):
    P = np.full((len(nodes),)*2, np.inf)
    for i,j in pairwise(path):
        P[i,j] = 0
    plot(nodes, P, F)


def main():
    global nodes, bps, fasttravel, caves, A, F, D, P, C

    with open('waypoints', 'r') as f:
        lines = f.readlines()
    lines = iter(lines)
    next(lines)

    nodes = {}
    for part in ['caves', 'towns', 'respawns', 'extra']:
        nodes[part] = {}
        for line in lines:
            if line.startswith('#'):
                break
            name, xyz, *_ = re.split(r'  +', line)
            nodes[part][name] = eval(f'[{xyz}]')

    bps = int(re.split(r'  +', next(lines))[1])
    fasttravel = {}
    for line in lines:
        name, duration = re.split(r'  +', line)
        fasttravel[name] = bps * int(duration)

    nodes = [NodeArray(nodes[part], name=part) for part in nodes]
    nodes = reduce(lambda x,y: x | y, nodes)

    A = adjacency_matrix(nodes)
    F = fasttravel_matrix(nodes)

    _ = np.full_like(A, np.inf)
    _[F < A] = F[F < A]
    assert np.array_equal(_, F), 'Some Fast Travel options are suboptimal'

    A[F < A] = F[F < A]
    D, P = floyd_warshall(A, return_predecessors=True)

    caves = nodes.parts['caves']
    C = D[caves][:,caves]


if __name__ == '__main__':
    main()
