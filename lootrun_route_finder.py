from functools import reduce, cache
from itertools import pairwise
from collections.abc import Sequence, Callable
from time import perf_counter
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go


BoolMask = Sequence[bool] | np.ndarray[bool] | bool
IndexArray = Sequence[int] | np.ndarray[int] | int
ArrayLike = np.ndarray | pd.DataFrame | pd.Series | Sequence


class TSP:
    """
    Inspired by https://github.com/fillipe-gsm/python-tsp

    Optimization:
        1. frozenset N is changed to an integer bitmask.
        2. Removed costs list and only store the current best value.
        3. Skip computing dist if D[ni,nj] is larger than current best cost.
           (only valid for non negative distances)

    Specialization:
        1. Scrolls allows travel from any point to a set of points called towns
           with a constant time. But there is a limit of 3 scroll charges.
        2. TODO: a scroll recharges after 10 minutes.
    """
    def __init__(self, distance_matrix: ArrayLike, scrolls=0):
        self.dist = cache(self._dist)
        # scroll dummy node
        self.nscroll = len(distance_matrix) - 1
        self.scrolls = scrolls
        # handling duplicate nodes
        kwargs = dict(return_index=True, return_inverse=True)
        _, ix, iv = np.unique(distance_matrix[:-1,:-1], axis=0, **kwargs)
        _, ixT, ivT = np.unique(distance_matrix[:-1,:-1], axis=1, **kwargs)
        self.duplicates = (ix[iv,None] == ix[iv]) & (ixT[ivT,None] == ixT[ivT])
        self.unique_inv = np.argmax(self.duplicates, axis=1)
        # adding a dummy node to convert path problem to cycle problem
        self.D = np.pad(distance_matrix, (0,1), constant_values=0)

    def _dist(self, ni: int, N: int, n0: int, scrolls: int) \
            -> tuple[int, float]:
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
        return nmin, costmin

    def solve(self, subset: IndexArray|BoolMask = True, start: int = None,
              *, loop=False) -> tuple[list, float]:
        bool_subset = np.zeros(len(self.duplicates), dtype=bool)
        bool_subset[subset] = 1
        duplic_subset = self.duplicates & bool_subset
        subset = self.unique_inv[subset]  # subset of unique nodes

        if not loop and start is None:
            ni = n0 = len(self.D) - 1  # start and stop at the dummy node
            solution = []
        else:
            ni = subset[0] if start is None else int(start)
            n0 = ni if loop else len(self.D) - 1  # stop at dummy if not loop
            solution = [ni]
        N = reduce(lambda a, b: a | 1 << b, subset, 0)
        N = N & ~(1 << ni) & ~(1 << len(self.D)-1)

        # Step 1: get minimum distance
        scrolls = self.scrolls
        best_distance = float(self.dist(ni, N, n0, scrolls)[1])

        # Step 2: get path with the minimum distance
        while N:
            ni = self.dist(ni, N, n0, scrolls)[0]
            solution.extend(np.nonzero(duplic_subset[ni])[0].tolist())
            N &= ~(1 << ni)
            if ni == self.nscroll:
                scrolls -= 1
        if loop:
            solution.append(int(solution[0]))

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
    WaypointGraph(waypoints: DataFrame,
                  bps=18,
                  fast_travel=True,
                  slash_kill=False,
                  teleport_scrolls=0)
        waypoints - a DataFrame containing waypoints and their info.
            (see waypoints.csv)
        bps - the speed of the player in blocks per second.
        fast_travel, slash_kill, teleport_scrolls - see below.

    Properties:
        waypoints - a copy of the waypoints DataFrame (read from waypoints.csv)
        distance_matrix - shortest path length between each two waypoints
        predecessor - the predecessor matrix to reconstruct the shortest path
        travel_type - a matrix storing the means of travel between two waypoints
        fast_travel - whether fast travel is allowed
        slash_kill - whether /kill is allowed
        teleport_scrolls - number of tp scrolls allowed to use

    Methods:
        enable(fast_travel: bool = None,
               slash_kill: bool = None,
               teleport_scrolls: bool = None)
            - update matrices according to enabled travel means.
        expand(path: list)
            - replaces each pair of points in the path with the
              shortest path between them.
        length(path: list)
            - returns the length of the path.
        find_route_between(subset: array[int | bool],
                           start: int = None,
                           loop: bool = False)
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
            - loop: if True searches for the shortest cycle
              aka. traveling saleman problem.
    """
    # all distances between waypoints assume the player can fly over trees and obstables
    # travel types:
    FLIGHT = 0
    FAST_TRAVEL = 1
    SLASH_KILL = 2
    TELEPORT_SCROLL = 3
    BLOCKED = 4

    def __init__(self, waypoints: pd.DataFrame, bps=18,
                 fast_travel=True, slash_kill=False, teleport_scrolls=0):
        self.waypoints = waypoints.copy()
        self.distance_matrix = None
        self.predecessor = None  # predecessor matrix from floyd-warshall
        self.travel_type = None

        self._EMPTYMATRIX = np.full((len(self.waypoints),) * 2, np.inf)
        # fast travel durations
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

        # distances between waypoints that are not blocked by mountains and barriers
        self._flight_path = self._build_map()
        self._fast_travel = self._build_fast_travel()
        self._slash_kill = self._build_slash_kill()
        self._ft = fast_travel
        self._sk = slash_kill
        self._tp = teleport_scrolls

        self.enable(fast_travel=self._ft,
                    slash_kill=self._sk,
                    teleport_scrolls=self._tp)

    def enable(self, fast_travel: bool = None, slash_kill: bool = None,
               teleport_scrolls: int = None):
        if fast_travel is not None:
            self._ft = fast_travel
        if slash_kill is not None:
            self._sk = slash_kill
        if teleport_scrolls is not None:
            self._tp = teleport_scrolls
        # choose minimum between map matrix, fast travel matrix, /kill matrix
        D = np.stack([self._flight_path,
            self._fast_travel if self._ft else self._EMPTYMATRIX,
            self._slash_kill if self._sk else self._EMPTYMATRIX])
        A = np.argmin(D, axis=0)
        D = np.take_along_axis(D, A[None], axis=0)[0]  # == np.min(D, axis=0)
        A[np.isinf(self._flight_path) & (A == 0)] = self.BLOCKED
        # add teleport scroll dummy node
        D = np.pad(D, (0, 1), constant_values=np.inf)
        A = np.pad(A, (0, 1), constant_values=self.BLOCKED)
        D[-1,*np.ix_(self.waypoints['Scroll'] == 1)] = 0
        A[-1,*np.ix_(self.waypoints['Scroll'] == 1)] = self.TELEPORT_SCROLL
        # not connecting other points to scroll dummy node
        # so floyd-warshall won't bridge over it
        D, P = floyd_warshall(D, return_predecessors=True)
        # adding edges to scroll node after computing bridges
        D[:-1,-1] = self._SCROLL
        P[:-1,-1] = np.arange(len(P) - 1)
        self.distance_matrix = D
        self.predecessor = P
        self.travel_type = A
        self.tsp = TSP(D, scrolls=self._tp)

    @property
    def fast_travel(self):
        return self._ft

    @property
    def slash_kill(self):
        return self._sk

    @property
    def teleport_scrolls(self):
        return self._tp

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

    def find_route_between(self, subset: IndexArray|BoolMask, start: int = None,
                           *, loop=False) -> pd.DataFrame:
        t = perf_counter()
        order, distance = self.tsp.solve(subset, start, loop=loop)
        expanded = self.expand(order)
        print(f'TSP solver took {perf_counter() - t:.0f}s')
        print(f'Total distance: {distance:.0f}')

        expanded_df = []
        for prev_waypoint, waypoint in pairwise([expanded[0]] + expanded):
            travel_type = self.travel_type[prev_waypoint, waypoint]
            if self.BLOCKED == travel_type:
                continue

            waypoint_df = self.waypoints.loc[[waypoint]]
            waypoint_df['Info'] = np.where(waypoint_df['Town'], 'Town',
                                  np.where(waypoint_df['Cave'], 'Cave', ''))

            if self.TELEPORT_SCROLL == travel_type:
                assert self.waypoints.loc[waypoint, 'Scroll'] == 1, \
                        f'{self.waypoints.loc[waypoint, "Name"]} ' \
                        f'doesn\'t have a teleport scroll'
                waypoint_df['Travel'] = 'teleport scroll'
            elif self.SLASH_KILL == travel_type:
                waypoint_df['Travel'] = '/kill'
            elif self.FAST_TRAVEL == travel_type:
                waypoint_df['Travel'] = 'fast travel'
            elif self.FLIGHT == travel_type:
                waypoint_df['Travel'] = ''

            expanded_df.append(waypoint_df)

        expanded_df = pd.concat(expanded_df, copy=False, ignore_index=True)
        expanded_df = expanded_df['Name Travel X Y Z Level Info'.split()]
        return expanded_df

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
    wp = pd.read_csv('waypoints.csv', skipinitialspace=True)
    wg = WaypointGraph(wp, bps=18,
                       fast_travel=True,
                       slash_kill=True,
                       teleport_scrolls=0)
    # find shortest route between caves from levels 72 to 80
    caves72_80 = (72 <= wp.Level) & (wp.Level <= 80) & (wp.Cave == 1)
    # wg.find_route_between(caves72_80)
    # takes about 60s
    # expected total distance 6405


if __name__ == '__main__':
    main()
