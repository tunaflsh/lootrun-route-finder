from functools import reduce, cache
from itertools import pairwise
from collections.abc import Sequence, Callable
from time import perf_counter
import csv
import numpy as np
from PIL import Image
import plotly.graph_objects as go


class TSP:
    """
    Source: https://github.com/fillipe-gsm/python-tsp
    """
    def __init__(self, distance_matrix: np.ndarray, scrolls=0):
        # adding a dummy node to convert path problem to cycle problem
        self._D = np.pad(distance_matrix, [(0,1), (0,1)], constant_values=0)
        self.dist = cache(self._dist)
        self.set_scrolls(scrolls)

    def set_scrolls(self, n):
        self._scrolls = n
        self._scroll_id = len(self._D) - 2

    def _dist(self, ni: int, N: int, n0: int, scrolls: int) \
            -> tuple[int, float]:
        """
        N - a bitmask representing a subset of nodes
        """
        costs = []
        if not N:
            costs.append((n0, self._D[ni,n0]))
        if scrolls > 0 and ni != self._scroll_id:
            nj = self._scroll_id
            costs.append((nj, self._D[ni,nj]
                          + self.dist(nj, N, n0, scrolls - 1)[1]))
        nj = 0
        while N >> nj:
            if N & 1 << nj:
                costs.append((
                    nj, self._D[ni,nj]
                    + self.dist(nj, N & ~(1 << nj), n0, scrolls)[1]))
            nj += 1
        return min(costs, key=lambda x: x[1])

    def solve(self, subset: Sequence | np.ndarray = None, start: int = None,
              *, loop=False) -> tuple[list, float]:
        n = self._D.shape[0]

        if subset is None:
            subset = np.arange(n - 2)
        subset, = np.ix_(subset)

        if not loop and start is None:
            ni = n0 = n - 1  # start and stop at the dummy node
            solution = []
        else:
            ni = subset[0] if start is None else int(start)
            n0 = ni if loop else n - 1  # stop at dummy if not loop
            solution = [ni]
        N = reduce(lambda a, b: a | 1 << b, subset, 0)
        N = N & ~(1<<ni) & ~(1<<n-1)

        # Step 1: get minimum distance
        scrolls = self._scrolls
        best_distance = float(self.dist(ni, N, n0, scrolls)[1])

        # Step 2: get path with the minimum distance
        while N:
            ni = int(self.dist(ni, N, n0, scrolls)[0])
            solution.append(ni)
            N &= ~(1<<ni)
            if ni == self._scroll_id:
                scrolls -= 1
        if loop:
            solution.append(int(solution[0]))

        return solution, best_distance


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


class Waypoints:
    """
    Stores info from the waypoints table like "waypoints.csv"

    Waypoints.i
        an array of ids. Each record has a unique id,
        which stays the same regardless of the order they are in the table.
    Waypoints.name
        an array of waypoint names.
    Waypoints.xyz
        an array of waypoint X, Y, Z coordinates.
    Waypoints.cave, Waypoints.town, Waypoints.scroll
        an arrays of ids of waypoints that are caves, towns or have
        teleportation scrolls.
    Waypoints.level
        an array of waypoint level requirements to access it.
    Waypoints.area
        an array of area names the waypoints are located in.

    Waypoints[name]
        to filter by name.
    Waypoints[idx]
        to access records by indexing or using boolean mask like Numpy.
        Note that indexing relies on the order of the records in the table
        and don't have the same meaning as Waypoints.i ids.
    Waypoints == area
        filter by area.
    Waypoints <, <=, ==, >=, > level
        filter by level.
    """
    def __init__(self, *, waypoints: 'Waypoints' = None,
                          data: dict[str, Sequence] = None,
                          filename: str = None):
        if waypoints is not None:
            self.__dict__.update(waypoints.__dict__)
            return

        if data is not None:
            self.__dict__.update(data)
            self._data = data
            self._namedict = {n: i for i, n in enumerate(self.name)}
            return

        if filename is not None:
            with open(filename, newline='') as file:
                reader = csv.reader(file, skipinitialspace=True)
                header = next(reader)
                data = list(zip(*list(reader)))

            data7 = [_.split('-') for _ in data[7]]
            self._data = {
                'i': np.arange(len(data[0])),
                'name': np.array(data[0]),
                'xyz': np.vectorize(float)(data[1:4]).T,
                '_cave': np.vectorize(int)(data[4]).astype(bool),
                '_town': np.vectorize(int)(data[5]).astype(bool),
                '_scroll': np.vectorize(int)(data[6]).astype(bool),
                'level': np.full((len(data7), max(map(len, data7))), np.nan),
                'area': np.array(data[8]),
            }
            self.__dict__.update(self._data)
            for i, u in enumerate(data7):
                for j, v in enumerate(u):
                    self.level[i,j] = int(v)
            self._namedict = {n: i for i, n in enumerate(data[0])}
            return

        raise ValueError('Either waypoints, data or filename must be specified.')

    @property
    def cave(self):
        return self[self._cave].i

    @property
    def town(self):
        return self[self._town].i

    @property
    def scroll(self):
        return self[self._scroll].i

    def __getitem__(self, name):
        """
        Access waypoints by name, index or boolean array.
        Works like numpy array indexing.
        """
        if isinstance(name, slice) or (isinstance(name, np.ndarray)
                                       and name.dtype == bool):
            idx = name
        elif isinstance(name, str):
            idx = [self._namedict[name]]
        elif isinstance(name, int):
            idx = [name]
        else:
            idx = [self._namedict[_] if isinstance(_, str) else _
                   for _ in dict.fromkeys(name)]
        data = {attr: data[idx] for attr, data in self._data.items()}
        return Waypoints(data=data)

    def __len__(self):
        return len(self._namedict)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __add__(self, other):
        overlap = np.isin(other.i, self.i)
        data = {attr: np.concat((data1, data2[~overlap]))
                for (attr, data1), data2 in zip(self._data.items(),
                                                other._data.values())}
        return Waypoints(data=data)

    def __sub__(self, other):
        overlap = np.isin(self.i, other.i)
        data = {attr: data[~overlap] for attr, data in self._data.items()}
        return Waypoints(data=data)

    def __and__(self, other):
        overlap = np.isin(self.i, other.i)
        data = {attr: data[overlap] for attr, data in self._data.items()}
        return Waypoints(data=data)

    def __or__(self, other):
        return self + other

    def __eq__(self, other: str | int | Sequence[str | int]):
        """
        Filter by name or level.
        """
        if isinstance(other, str) or isinstance(other, Sequence) \
                and all(isinstance(_, str) for _ in other):
            return self[np.isin(self.area, other)]
        return self[np.isin(self.level, other).any(axis=1)]

    def __ne__(self, other: str | int | Sequence[str | int]):
        """
        Filter by name or level.
        """
        if isinstance(other, str) or isinstance(other, Sequence) \
                and all(isinstance(_, str) for _ in other):
            return self[~np.isin(self.area, other)]
        return self[~np.isin(self.level, other).any(axis=1)]

    def __lt__(self, other: int):
        """
        Filter by level.
        """
        return self[(self.level < other).any(axis=1)]

    def __gt__(self, other: int):
        """
        Filter by level.
        """
        return self[(self.level > other).any(axis=1)]

    def __le__(self, other: int):
        return (self < other) | (self == other)

    def __ge__(self, other: int):
        return (self > other) | (self == other)

    def __repr__(self):
        maxname = min(18, max(map(len, self.name)))
        maxarea = min(18, max(map(len, self.area)))
        s = [f'       {"Name":>{maxname}}         X    Y    Z    '
             f'Cave|Town|Scroll  Level     {"Area":>{maxarea}}',
             '-' * (maxname + maxarea + 59),]
        for record in self:
            i, name, (x,y,z), *cts, level, area = \
                    map(lambda v: v[0], list(record._data.values()))
            s.append(f'{i:>3}    '
                     f'{self._truncate(name, limit=maxname)}    '
                     f'{x:6.0f} {y:4.0f} {z:6.0f}    '
                     f'{"    ".join(map(str, map(int, cts)))}    '
                     f'{"-".join(str(int(_)) for _ in level if ~np.isnan(_)):^7}    '
                     f'{self._truncate(area, limit=maxarea)}')
        return '\n'.join(s)

    def _ipython_key_completions_(self):
        return self.name.tolist()

    @staticmethod
    def _truncate(*value, limit=None):
        value = ' '.join(map(str, value))
        if limit is None:
            return value
        if len(value) <= limit:
            return f'{value:>{limit}}'
        return f'{value[:limit-3]:>{limit-3}}...'

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text('...')
            return
        p.text(str(self[:20]))
        if 20 < len(self):
            maxname = min(18, max(map(len, self[:20].name)))
            maxarea = min(18, max(map(len, self[:20].area)))
            p.text(f'\n  :    {":":>{maxname}}    {":":>6} {":":>4} '
                   f'{":":>6}    :    :    :   {":":^9}   {":":>{maxarea}}')


class WaypointGraph(Waypoints):
    _BPS = 18
    _TUNNEL = 0.1 * _BPS
    _VSS = 42 * _BPS
    _OBELISK = 9 * _BPS
    _NEXUS_HUB = 10 * _BPS
    _AIRSHIP = 7 * _BPS
    _JUGGLER = 20 * _BPS
    _LIGHT_PORTAL_IN = 10 * _BPS
    _LIGHT_PORTAL_OUT = 20 * _BPS
    _SLASH_KILL = 3 * _BPS
    _SCROLL = 7 * _BPS

    def __init__(self, waypoints: Waypoints, map_image_path: str = None):
        super().__init__(waypoints=waypoints)
        # adjacency matrix based on wynncraft map
        self._W = np.full((len(self),) * 2, np.inf)
        # fast travel matrix
        self._F = np.full_like(self._W, np.inf)
        # /kill matrix
        self._K = np.full_like(self._W, np.inf)
        # scroll usage limit
        self._S = 0
        # build matrices
        self.build_wynncraft_map()
        self.enable_fast_travel()
        # self.enable_slash_kill()
        # self.enable_scrolls()
        # background map image
        self._map = Image.open(map_image_path)

    @staticmethod
    def update(matrix_builder: Callable[['WaypointGraph', ...], None]) \
            -> Callable[['WaypointGraph'], None]:
        def build(self, *args, **kwargs):
            matrix_builder(self, *args, **kwargs)
            # wynncraft map matrix, fast travel matrix, /kill matrix
            WFK = np.stack((self._W, self._F, self._K))
            WFK = np.min(WFK, axis=0)
            # scroll teleportation
            WFKS = np.pad(WFK, (0, 1), constant_values=np.inf)
            WFKS[-1,self.scroll] = 0
            # not setting path to scroll dummy node
            # so floyd_warshall won't bridge over it
            D, P = floyd_warshall(WFKS, return_predecessors=True)
            # setting path to scroll after computing bridges
            D[:-1,-1] = self._SCROLL
            P[:-1,-1] = np.arange(len(P) - 1)
            self._D = D
            self._P = P
            self.tsp = TSP(D, scrolls=self._S)

        return build

    @staticmethod
    def _distance_metric(src: Waypoints, dst: Waypoints) -> np.ndarray:
        src, dst = src.xyz, dst.xyz
        if src.ndim == 1:
            src = src[None]
        D = dst[None] - src[:,None]
        D[D[...,1] < 0, 1] = 0  # ignore falling distance
        D = np.sum(D**2, axis=-1)**0.5
        return D

    def _disconnect(self, W: np.ndarray, U: Waypoints, V: Waypoints = None):
        V = V or self
        W[np.ix_(U.i, (V - U).i)] = np.inf
        W[np.ix_((V - U).i, U.i)] = np.inf

    def _connect(self, W: np.ndarray, U: Waypoints, V: Waypoints = None):
        V = V or U
        W[np.ix_(U.i, V.i)] = self._distance_metric(U, V)
        W[np.ix_(V.i, U.i)] = self._distance_metric(V, U)

    @update
    def build_wynncraft_map(self):
        W = np.full((len(self),) * 2, np.inf)
        # Ocean - Wynn - Nesaak - Gavel
        ocean_exposure = self == ['Wynn Plains', 'Nesaak Tundra', 'Ocean']
        ocean_exposure += self['Llevigar', 'Weird Wilds', 'Jofash Dock']
        self._connect(W, ocean_exposure)
        self._disconnect(W, self['Icy Vigil'],
                         self['The Passage', 'Ragni', 'Ragni-Detlas Tunnel'])
        self._disconnect(W, self['Llevigar'],
                         self['Icy Vigil', 'Ragni', 'The Passage'])
        # Gavel
        gavel_area = self == ['Llevigar Plains', 'Kander Forest', 'Dark Forest',
                              'Pre-Light Forest', 'Light Forest', 'Aldorei',
                              'Gylia Plains', 'Canyon of the Lost', 'Olux Swamp']
        self._connect(W, gavel_area)
        self._connect(W, *self['Llevigar', 'Llevigar Gate'])
        # Jungle
        self._connect(W, self == 'Troms Jungle')
        self._connect(W, *self['The Passage', 'Troms'])
        self._connect(W, *self['The Great Bridge', 'Icy Vigil'])
        self._connect(W, *self['Dernel Tunnel', 'Basalt Basin'])
        # The Realm of Light
        self._connect(W, self == 'The Realm of Light')
        # Molten Heights
        self._connect(W, (self == 'Molten Heights') + self['Sky-Molten Tunnel'])
        self._connect(W, *self['Molten Heights Gate', 'Molten Heights Lava Lake'])
        # Sky Islands
        self._connect(W, (self == 'Sky Islands') + self['Thesead-Eltom Tunnel'])
        self._connect(W, *self['Jofash Tunnel', 'Jofash Dock'])

        self._W = W

    @update
    def enable_fast_travel(self):
        F = np.full((len(self),) * 2, np.inf)
        # Ragni-Detlas Tunnel
        ra, de = self['Ragni-Detlas Tunnel', 'Detlas'].i
        F[ra, de] = F[de, ra] = self._TUNNEL
        # V.S.S. Seaskipper
        nr, ns, se, ll, jd = self['Nemract', 'Nesaak', 'Selchar', 'Llevigar',
                                  'Jofash Dock'].i
        F[nr, [ns, se, ll]] = self._VSS
        F[ll, [ns, se, nr, jd]] = self._VSS
        F[jd, [nr, se, ll]] = self._VSS
        F[se, [nr, ll, ns, jd]] = self._VSS
        F[ns, [nr, se, ll]] = self._VSS
        # Mysterious Obelisk
        tt = self['Tempo Town'].i
        F[nr, tt] = F[tt, nr] = self._OBELISK
        # Nexus Hub
        hub = np.zeros_like(F, dtype=bool)
        hub[[nr, se, ll]] = 1
        hub = hub & hub.T
        np.fill_diagonal(hub, 0)
        assert (self._NEXUS_HUB < F[hub]).all(), \
                'Nexus Hub is not faster than _V.S.S.'
        F[hub & hub.T] = self._NEXUS_HUB
        # Calo's Airship
        la = self['Letvus Airbase'].i
        F[de, la] = F[la, de] = self._AIRSHIP
        # The Juggler's Tavern
        ci, at, th, ro = self['Cinfras', 'Aldorei Town', 'Thesead', 'Rodoroc'].i
        F[ci, [at, th, ro]] = self._JUGGLER
        # Nexus Gate
        ls, rl = self['Light\'s Secret', 'Light Portal'].i
        F[ls, rl] = self._LIGHT_PORTAL_IN
        F[rl, ls] = self._LIGHT_PORTAL_OUT
        # Eltom-Thesead Tunnel
        el, th = self['Eltom', 'Thesead-Eltom Tunnel'].i
        F[el, th] = F[th, el] = self._TUNNEL
        # Dwarven Trading Tunnel
        tn = self['Thanos'].i
        F[tn, ro] = F[ro, tn] = self._TUNNEL
        # Colossus Tunnel
        ct, kb = self['Colossus Tunnel', 'Kandon-Beda'].i
        F[ct, kb] = F[kb, ct] = self._TUNNEL

        self._F = F

    @update
    def disable_fast_travel(self):
        self._F = np.full((len(self),) * 2, np.inf)

    @update
    def enable_slash_kill(self):
        V = self != 'The Realm of Light'
        D = self._distance_metric(V, V[V.town])
        T = V[V.town].i[np.argmin(D, axis=1)]
        K = np.full((len(self),) * 2, np.inf)
        K[V.i, T] = self._SLASH_KILL
        self._K = K

    @update
    def disable_slash_kill(self):
        self._K = np.full((len(self),) * 2, np.inf)

    def enable_scrolls(self, n=3):
        self._S = n
        self.tsp.set_scrolls(n)

    def disable_scrolls(self):
        self._S = 0
        self.tsp.set_scrolls(0)

    def find_route_between(self, idx: Sequence[int] | np.ndarray[int],
                           start: int = None, *, loop=False):
        t = perf_counter()
        order, distance = self.tsp.solve(subset=idx, start=start, loop=loop)
        print(f'TSP solver took {perf_counter() - t:.0f}s')
        print(f'Total distance: {distance:.0f}')
        expanded = self.expand_path(order)
        print(expanded)
        print(self.name[[i for i in expanded if i < len(self) - self._S]])
        self.plot(expanded)

    def expand_path(self, path: list[int]) -> list[int]:
        expanded = []
        j = path[-1]
        i = -2
        while -len(path) <= i:
            expanded.append(j)
            if path[i] == (j := int(self._P[path[i], j])):
                i -= 1
        expanded.append(path[i + 1])
        return expanded[::-1]

    def path_length(self, path: list[int]) -> float:
        return reduce(lambda l,e: l + self._D[e], pairwise(path), 0)

    def plot(self, path: list[int] = None):
        argmin = np.argmin(np.stack((self._W, self._F, self._K)), axis=0)
        argmin[np.isinf(self._W) & (argmin == 0)] = 4

        #        W   F   K   S
        lines = [[], [], [], []]

        xz = np.concat((self.xyz[:,[0,2]], [[np.nan] * 2]))
        if path:
            for i, j in pairwise(path):
                if j == len(self):
                    lines[3].append(i)
                    continue
                if i == len(self):
                    lines[3].extend([j, -1])
                    continue
                a = argmin[i,j]
                if not lines[a]:
                    lines[a].extend([i, j])
                elif i == lines[a][-1]:
                    lines[a].append(j)
                else:
                    lines[a].extend([-1, i, j])
            lines = [xz[line].T for line in lines]
        else:
            for i in range(3):
                line = np.unique(np.sort(np.nonzero(argmin == i), axis=0), axis=1)
                line = np.concat((line, -np.ones((1, line.shape[1]), dtype=int)))
                line = xz[line.T.flatten()]
                lines[i] = line.T

        fig = go.Figure(layout=go.Layout(template='plotly_dark'))
        for line, dash, color in zip(
                lines,
                ['dot'] + ['solid'] * 3,
                ['turquoise', 'limegreen', 'crimson', 'darkorange']):
            fig.add_trace(go.Scatter(
                {k: v for k, v in zip('xy', line)},
                hoverinfo='skip',
                showlegend=False,
                line=go.scatter.Line(color=color, dash=dash),
                mode='lines'))
        fig.add_trace(go.Scatter(
            {'text': self.name,
             'x': self.xyz[:,0],
             'y': self.xyz[:,2]},
            customdata=self.xyz,
            hovertemplate='%{text}<br>'
                          '%{customdata[0]} '
                          '%{customdata[1]} '
                          '%{customdata[2]}'
                          '<extra></extra>',
            showlegend=False,
            marker=go.scatter.Marker(color='orange'),
            mode='markers'))
        fig.update_yaxes(scaleanchor='x', autorange='reversed')
        fig.add_layout_image(
                source=self._map, layer='below',
                xref='x', yref='y', x=-2392, y=-6607,
                sizex=self._map.size[0], sizey=self._map.size[1])
        fig.show()


def main():
    global wp, wg
    wp = Waypoints(filename='waypoints.csv')
    wg = WaypointGraph(wp, map_image_path='TopographicMap.png')
    wg.enable_slash_kill()
    wg.enable_scrolls(n=3)
    # find shortest route between caves from level 72 to 80
    wg.find_route_between((72 <= wg <= 80).cave)


if __name__ == '__main__':
    main()
