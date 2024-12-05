from datetime import timedelta
from functools import reduce
from itertools import pairwise
from collections.abc import Sequence
from time import perf_counter
import os
from msgspec import msgpack
import numpy as np
import pandas as pd


BoolMask = Sequence[bool] | np.ndarray[bool] | bool
IndexArray = Sequence[int] | np.ndarray[int] | int
ArrayLike = np.ndarray | pd.DataFrame | pd.Series | Sequence

encoder = msgpack.Encoder()


class TSP:
    """
    Inspired by https://github.com/fillipe-gsm/python-tsp

    Optimization:
        1. frozenset N is changed to an integer bitmask.
        2. Removed costs list and only store the current best value.
        3. Skip computing dist if D[ni,nj] is larger than current best cost.
           (only valid for non negative distances)
        4. Custom caching. Storing and loading from the disk.

    Specialization:
        1. Scrolls allows travel from any point to a set of points called towns
           with a constant time. But there is a limit of 3 scroll charges.
        2. TODO: a scroll recharges after 10 minutes.
    """
    def __init__(self, distance_matrix: ArrayLike, scrolls=0, cycle=False,
                 cache_file: str = None):
        # scroll dummy node
        self.nscroll = len(distance_matrix) - 1
        self.scrolls = scrolls
        assert scrolls <= 3, f'Scroll capacity in Wynncraft is 3, but {scrolls=}'

        # tsp or shortest hamiltonian path
        self.cycle = cycle

        # loading cache
        self.memo = {}
        self.cache_file = cache_file

        # handling duplicate nodes
        kwargs = dict(return_index=True, return_inverse=True)
        _, ix, iv = np.unique(distance_matrix[:-1,:-1], axis=0, **kwargs)
        _, ixT, ivT = np.unique(distance_matrix[:-1,:-1], axis=1, **kwargs)
        self.duplicates = (ix[iv,None] == ix[iv]) & (ixT[ivT,None] == ixT[ivT])
        self.unique_inv = np.argmax(self.duplicates, axis=1)

        # adding a dummy node to convert path problem to cycle problem
        self.D = np.pad(distance_matrix, (0,1), constant_values=0)

    def dist(self, ni: int, N: int, n0: int, scrolls: int) -> tuple[int, float]:
        try:
            return self.memo[n0][scrolls][ni, N]
        except KeyError:
            pass
        nmin, costmin = None, np.inf
        if not N:
            nmin, costmin = n0, self.D[ni,n0]
        Dij = self.D[ni, self.nscroll]
        if scrolls > 0 and ni != self.nscroll and Dij < costmin:
            cost = Dij + self.dist(self.nscroll, N, n0, scrolls - 1)[1]
            if cost < costmin:
                nmin, costmin = self.nscroll, cost
        nj = 0
        while N >> nj:
            Dij = self.D[ni,nj]
            if N & 1 << nj and Dij < costmin:
                cost = Dij + self.dist(nj, N & ~(1 << nj), n0, scrolls)[1]
                if cost < costmin:
                    nmin, costmin = nj, cost
            nj += 1
        costmin = float(costmin)
        self.memo[n0][scrolls][ni, N] = nmin, costmin
        return nmin, costmin

    def solve(self, subset: IndexArray|BoolMask = True, start: int = None) -> tuple[list, float]:
        bool_subset = np.zeros(len(self.duplicates), dtype=bool)
        bool_subset[subset] = 1
        duplic_subset = self.duplicates & bool_subset
        subset = self.unique_inv[subset].tolist()  # subset of unique nodes

        if not self.cycle and start is None:
            ni = n0 = len(self.D) - 1  # start and stop at the dummy node
            solution = []
        else:
            ni = subset[0] if start is None else int(start)
            n0 = ni if self.cycle else len(self.D) - 1  # stop at dummy if not cycle
            solution = [ni]
        N = reduce(lambda a, b: a | 1 << b, subset, 0)
        N = N & ~(1 << ni) & ~(1 << len(self.D)-1)

        # Step 0: initialize memo
        if not self.memo and self.cache_file:
            try:
                t = perf_counter()
                with open(self.cache_file, 'rb') as cf:
                    self.memo = msgpack.decode(cf.read())
                print(f'Loading cache time: {perf_counter() - t:.0f}s')
            except FileNotFoundError:
                pass
        scrolls = self.scrolls
        if n0 not in self.memo:
            # for each scroll up to `scrolls` included
            self.memo[n0] = [{} for _ in range(3 + 1)]
        memosize = sum(map(len, self.memo[n0]))

        # Step 1: get minimum distance
        t = perf_counter()
        best_distance = self.dist(ni, N, n0, scrolls)[1]
        print(f'TSP runtime: {perf_counter() - t:.0f}s')
        print(f'Total distance: {best_distance:.0f}')

        # saving cache
        if memosize < sum(len(s) for s in self.memo[n0]) \
                and self.cache_file:
            t = perf_counter()
            dirname = os.path.dirname(self.cache_file)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(self.cache_file, 'wb') as cf:
                cf.write(encoder.encode(self.memo))
            print(f'Saving cache time: {perf_counter() - t:.0f}s')

        # Step 2: reconstructing the path
        while N:
            ni = self.dist(ni, N, n0, scrolls)[0]
            N &= ~(1 << ni)
            if ni == self.nscroll:
                scrolls -= 1
                solution.append(ni)
                continue
            solution.extend(np.nonzero(duplic_subset[ni])[0].tolist())
        if self.cycle:
            solution.append(solution[0])

        return solution, best_distance


def floyd_warshall(A: ArrayLike, return_predecessors=False):
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


class WaypointGraph:
    """
    Input parameters:
        waypoints - a copy of the waypoints DataFrame (read from waypoints.csv)
        bps - average moving speed of the player in blocks per second
        fast_travel - whether fast travel is allowed
        slash_kill - whether /kill is allowed
        scrolls - number of teleport scrolls allowed to use
        cycle - whether the path is open or closes on itself, i.e.
            TSP vs Shortest Hamiltonian Path
        cache_file_template - a string formatter operating on variables:
            bps, fast_travel, slash_kill, cycle.
            Example: 'bps={bps}ft={fast_travel}sk={slash_kill}cy={cycle}.msgspec'
            Note: `scrolls` is not used since different scroll number will
                  still have the same memo cache.

    Other parameters:
        distance_matrix - shortest path length between each two waypoints
        predecessor - the predecessor matrix to reconstruct the shortest path
        travel_type - a matrix storing the means of travel between two waypoints

    Methods:
        update(bps: int = None, fast_travel: bool = None, slash_kill: bool = None,
               scrolls: bool = None, cycle: bool = None, cache_file_template: str = None)
            - update parameters
        expand(path: list)
            - replaces each pair of points in the path with the
              shortest path between them.
        length(path: list)
            - returns the length of the path.
        find_route_between(subset: array[int | bool],
                           start: int = None,
                           cycle: bool = False)
            - returns the shortest path passing through all
              waypoints in the subset in the form of DataFrame.
              The dataframe has 7 columns:
                  Name - the name of the waypoint
                  Travel - the travel type
                  X Y Z - the coordinates
                  Level - the level requirement of the waypoint
                  Info - extra information about the waypoint
                      (e.g. Town/Cave)
            - subset: an index array-like for the waypoints
              that will be passed through.
            - start: if given is the index of the starting point.
            - cycle: if True searches for the shortest cycle
              aka. traveling saleman problem.
    """
    # all distances between waypoints assume the player can fly over trees and obstables
    # travel types:
    FLIGHT = 0
    FAST_TRAVEL = 1
    SLASH_KILL = 2
    SCROLL = 3
    BLOCKED = 4

    def __init__(self, waypoints: pd.DataFrame, bps=18, cache_file_template: str = None,
                 fast_travel=True, slash_kill=False, scrolls=0, cycle=False):
        self.waypoints = waypoints.copy()
        self.distance_matrix = None
        self.predecessor = None  # predecessor matrix from floyd-warshall
        self.travel_type = None
        self.tsp = None  # tsp solver

        self._EMPTYMATRIX = np.full((len(self.waypoints),) * 2, np.inf)
        # fast travel durations
        self.bps = bps
        self._TUNNEL = 0.1 * bps
        self._VSS = 42 * bps
        self._OBELISK = 9 * bps
        self._NEXUS_HUB = 10 * bps
        self._AIRSHIP = 7 * bps
        self._JUGGLER = 20 * bps
        self._LIGHT_PORTAL_IN = 10 * bps
        self._LIGHT_PORTAL_OUT = 20 * bps
        self._SLASH_KILL = 3 * bps
        self._SCROLL = 7 * bps

        # wynncraft map adjacency/distance matrix
        self._flight_path = self._build_map()
        self._fast_travel = self._EMPTYMATRIX
        self._slash_kill = self._EMPTYMATRIX

        self.fast_travel = None
        self.slash_kill = None
        self.scrolls = None
        self.cycle = None
        self.cache_file_template = None

        self.update(bps=bps, fast_travel=fast_travel, slash_kill=slash_kill,
                    scrolls=scrolls, cycle=cycle,
                    cache_file_template=cache_file_template)

    def update(self, bps: int = None, fast_travel: bool = None, slash_kill: bool = None,
               scrolls: int = None, cycle: bool = None, cache_file_template: str = None):
        should_update_fast_travel = False
        should_update_slash_kill = False
        should_update_tsp = False

        if bps is not None \
                and bps != self.bps \
                or self.bps is None:
            self.bps = bps
            should_update_fast_travel = True
            should_update_slash_kill = True
            should_update_tsp = True
        if fast_travel is not None \
                and fast_travel != self.fast_travel \
                or self.fast_travel is None:
            self.fast_travel = fast_travel
            should_update_fast_travel = True
            should_update_tsp = True
        if slash_kill is not None \
                and slash_kill != self.slash_kill \
                or self.slash_kill is None:
            self.slash_kill = slash_kill
            should_update_slash_kill = True
            should_update_tsp = True
        if scrolls is not None \
                and scrolls != self.scrolls \
                or self.scrolls is None:
            self.scrolls = scrolls
            should_update_tsp = True
        if cycle is not None \
                and cycle != self.cycle \
                or self.cycle is None:
            self.cycle = cycle
            should_update_tsp = True
        if cache_file_template is not None \
                and cache_file_template != self.cache_file_template \
                or self.cache_file_template is None:
            self.cache_file_template = cache_file_template
            should_update_tsp = True

        if should_update_fast_travel:
            self._fast_travel = self._build_fast_travel() if self.fast_travel \
                                else self._EMPTYMATRIX
        if should_update_slash_kill:
            self._slash_kill = self._build_slash_kill() if self.slash_kill \
                               else self._EMPTYMATRIX

        cache_file = None
        if self.cache_file_template:
            cache_file = self.cache_file_template.format(
                bps=self.bps,
                fast_travel=int(self.fast_travel),
                slash_kill=int(self.slash_kill),
                cycle=int(self.cycle))

        if should_update_fast_travel | should_update_slash_kill:
            # choose minimum between map matrix, fast travel matrix, /kill matrix
            D = np.stack([self._flight_path, self._fast_travel, self._slash_kill])
            A = np.argmin(D, axis=0)
            D = np.take_along_axis(D, A[None], axis=0)[0]  # == np.min(D, axis=0)
            A[np.isinf(self._flight_path) & (A == 0)] = self.BLOCKED
            # add teleport scroll dummy node
            D = np.pad(D, (0, 1), constant_values=np.inf)
            A = np.pad(A, (0, 1), constant_values=self.BLOCKED)
            D[-1,*np.ix_(self.waypoints['Scroll'] == 1)] = 0
            A[-1,*np.ix_(self.waypoints['Scroll'] == 1)] = self.SCROLL
            # not connecting other points to scroll dummy node
            # so floyd-warshall won't bridge over it
            D, P = floyd_warshall(D, return_predecessors=True)
            # adding edges to scroll node after computing bridges
            D[:-1,-1] = self._SCROLL
            P[:-1,-1] = np.arange(len(P) - 1)
            self.distance_matrix = D
            self.predecessor = P
            self.travel_type = A

        if should_update_tsp:
            self.tsp = TSP(self.distance_matrix, scrolls=self.scrolls,
                           cycle=self.cycle, cache_file=cache_file)

    @staticmethod
    def _distance_metric(src: ArrayLike, dst: ArrayLike) -> np.ndarray:
        if src.ndim == 1:
            src = np.expand_dims(src, 0)
        D = np.expand_dims(dst, 0) - np.expand_dims(src, 1)
        D[D[...,1] < 0, 1] = 0  # ignore falling distance
        D = np.sum(D**2, axis=2)**0.5
        return D

    def _build_map(self) -> ArrayLike:
        def connect(U: BoolMask, V: BoolMask = None):
            # Connect all nodes from subset U to subset V and v.v.
            # If V is not provided, makes U a complete subgraph.
            if V is None:
                V = U
            Uxyz = self.waypoints.loc[U, list('XYZ')]
            Vxyz = self.waypoints.loc[V, list('XYZ')]
            D[np.ix_(U, V)] = self._distance_metric(Uxyz, Vxyz)
            D[np.ix_(V, U)] = self._distance_metric(Vxyz, Uxyz)

        def disconnect(U: BoolMask, V: BoolMask = True):
            # Disconnect node subset U from subset V-U if V is provided.
            # Disconnect U from the rest of the graph otherwise.
            D[np.ix_(U, V & ~U)] = np.inf
            D[np.ix_(V & ~U, U)] = np.inf

        D = np.copy(self._EMPTYMATRIX)
        area = self.waypoints['Area']
        name = self.waypoints['Name']
        # ocean clique
        connect(area.isin(['Wynn Plains', 'Nesaak Tundra', 'Ocean'])
                | (name == 'Weird Wilds'))
        disconnect(name.isin(['Icy Vigil', 'Ragni', 'The Passage']),
                   name == 'Llevigar Dock')
        # nesaak mountain ranges
        disconnect(name.isin(['The Passage', 'Ragni', 'Ragni-Detlas Tunnel']),
                   name == 'Icy Vigil')
        # gavel clique
        gavel = area.isin(['Llevigar Plains', 'Kander Forest', 'Dark Forest',
                           'Pre-Light Forest', 'Light Forest', 'Aldorei',
                           'Gylia Plains', 'Canyon of the Lost', 'Olux Swamp'])
        connect(gavel)
        connect(name == 'Llevigar Dock', name == 'Llevigar North')
        # Jungle
        connect(area == 'Troms Jungle')
        connect(name == 'The Passage', name == 'Troms')
        connect(name == 'The Great Bridge', name == 'Icy Vigil')
        connect(name == 'Dernel Tunnel', name == 'Basalt Basin')
        # The Realm of Light
        connect(area == 'The Realm of Light')
        # Molten Heights
        connect((area == 'Molten Heights') | (name == 'Sky-Molten Tunnel'))
        connect(name == 'Molten Heights Gate', name == 'Molten Heights Lava Lake')
        # Sky Islands
        connect((area == 'Sky Islands') | (name == 'Thesead-Eltom Tunnel'))
        connect(name == 'Jofash Tunnel', name == 'Jofash Dock')

        np.fill_diagonal(D, 0)
        return D

    def _build_fast_travel(self) -> ArrayLike:
        D = np.copy(self._EMPTYMATRIX)
        name = self.waypoints['Name']
        # Ragni-Detlas Tunnel
        ra, de = name == 'Ragni-Detlas Tunnel', name == 'Detlas'
        D[ra, de] = D[de, ra] = self._TUNNEL
        # V.S.S. Seaskipper
        nr, ns, se, ll, jd = np.array(name) == [['Nemract'], ['Nesaak'], ['Selchar'],
                                                ['Llevigar Dock'], ['Jofash Dock']]
        D[nr, ns|se|ll] = self._VSS
        D[ll, ns|se|nr|jd] = self._VSS
        D[jd, nr|se|ll] = self._VSS
        D[se, nr|ll|ns|jd] = self._VSS
        D[ns, nr|se|ll] = self._VSS
        # Mysterious Obelisk
        tt = name == 'Tempo Town'
        D[nr, tt] = D[tt, nr] = self._OBELISK
        # Nexus Hub
        hub = nr|se|ll
        hub = hub & hub[:,None]
        np.fill_diagonal(hub, 0)
        assert (self._NEXUS_HUB < D[hub]).all(), \
                'Nexus Hub is not faster than _V.S.S.'
        D[hub] = self._NEXUS_HUB
        # Calo's Airship
        la = name == 'Letvus Airbase'
        D[de, la] = D[la, de] = self._AIRSHIP
        # The Juggler's Tavern
        ci, at, th, ro = np.array(name) == [['Cinfras'], ['Aldorei Town'],
                                            ['Thesead'], ['Rodoroc']]
        D[ci, at|th|ro] = self._JUGGLER
        # Nexus Gate
        ls, rl = name == 'Light\'s Secret', name == 'Light Portal'
        D[ls, rl] = self._LIGHT_PORTAL_IN
        D[rl, ls] = self._LIGHT_PORTAL_OUT
        # Eltom-Thesead Tunnel
        el, th = name == 'Eltom', name == 'Thesead-Eltom Tunnel'
        D[el, th] = D[th, el] = self._TUNNEL
        # Dwarven Trading Tunnel
        tn = name == 'Thanos'
        D[tn, ro] = D[ro, tn] = self._TUNNEL
        # Colossus Tunnel
        ct, kb = name == 'Colossus Tunnel', name == 'Kandon-Beda'
        D[ct, kb] = D[kb, ct] = self._TUNNEL
        return D

    def _build_slash_kill(self) -> ArrayLike:
        inside_region = self.waypoints['Area'] != 'The Realm of Light'
        towns = inside_region & (self.waypoints['Town'] == 1)
        Vxyz = self.waypoints.loc[inside_region, list('XYZ')]
        towns_xyz = self.waypoints.loc[towns, list('XYZ')]
        D = self._distance_metric(Vxyz, towns_xyz)
        closest_town = np.nonzero(towns)[0][np.argmin(D, axis=1)]
        slash_kill = np.copy(self._EMPTYMATRIX)
        slash_kill[inside_region, closest_town] = self._SLASH_KILL
        return slash_kill

    def find_route_between(self, subset: IndexArray|BoolMask, start: int = None) -> pd.DataFrame:
        order, distance = self.tsp.solve(subset, start)
        duration = distance // self.bps
        print('Travel duration: {:.0f}m{:.0f}s'.format(*divmod(duration, 60)))
        order = self.expand(order)
        order_df = []
        for prev_waypoint, waypoint in pairwise([order[0]] + order):
            travel_type = self.travel_type[prev_waypoint, waypoint]
            if self.BLOCKED == travel_type:
                continue

            waypoint_df = self.waypoints.loc[[waypoint]]
            info = ['Town'] if waypoint_df['Town'].item() else \
                   ['Cave'] if waypoint_df['Cave'].item() else []

            if self.SCROLL == travel_type:
                assert self.waypoints.loc[waypoint, 'Scroll'] == 1, \
                        f'{self.waypoints.loc[waypoint, "Name"]} ' \
                        f'doesn\'t have a teleport scroll'
                waypoint_df['Travel'] = self.SCROLL
                waypoint_df['Info'] = ', '.join([*info, 'tp scroll'])
            elif self.SLASH_KILL == travel_type:
                waypoint_df['Travel'] = self.SLASH_KILL
                waypoint_df['Info'] = ', '.join([*info, '/kill'])
            elif self.FAST_TRAVEL == travel_type:
                waypoint_df['Travel'] = self.FAST_TRAVEL
                waypoint_df['Info'] = ', '.join([*info, 'fast travel'])
            elif self.FLIGHT == travel_type:
                waypoint_df['Travel'] = self.FLIGHT
                waypoint_df['Info'] = ', '.join([*info])

            order_df.append(waypoint_df)

        order_df = pd.concat(order_df, copy=False)
        order_df = order_df['Name Travel X Y Z Level Info'.split()]
        return order_df, distance

    def expand(self, path: list[int]) -> list[int]:
        expanded = []
        j = path[-1]
        i = -2
        while -len(path) <= i:
            expanded.append(j)
            if path[i] == (j := int(self.predecessor[path[i], j])):
                i -= 1
        expanded.append(path[i + 1])
        return expanded[::-1]

    def length(self, path: list[int]) -> float:
        return reduce(lambda l,e: l + self.distance_matrix[e], pairwise(path), 0)


def main():
    global wp, wg, caves72_80
    wp = pd.read_csv('assets/waypoints.csv', skipinitialspace=True)
    bps = 18
    ft = True
    sk = True
    sc = 3
    cy = False
    cf = '.cache/bps{bps}ft{fast_travel}sk{slash_kill}cy{cycle}.msgspec'
    wg = WaypointGraph(wp, bps=bps, fast_travel=ft, slash_kill=sk,
                       scrolls=sc, cycle=cy, cache_file_template=cf)
    # find shortest route between caves from levels 72 to 80
    caves72_80 = (72 <= wp.Level) & (wp.Level <= 80) & (wp.Cave == 1)
    # print(wg.find_route_between(caves72_80))
    # takes about 250s
    # expected total distance 4958


if __name__ == '__main__':
    main()
