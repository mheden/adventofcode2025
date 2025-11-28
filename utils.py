from contextlib import suppress
from queue import PriorityQueue, Empty
import os.path
import re
import time


BIGNUM = 10**100


class P2d:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def pos(self):
        return (self.x, self.y)

    def __repr__(self):
        return "P2d(%d, %d)" % (self.x, self.y)


class Rect:
    def __init__(self, tl: P2d, br: P2d):
        assert_ge(br.x, tl.x)
        assert_ge(br.y, tl.y)
        self.tl = tl
        self.br = br

    def overlap(self, other):
        return not (
            self.br.x < other.tl.x
            or self.tl.x > other.br.x
            or self.br.y < other.tl.y
            or self.tl.y > other.br.y
        )

    def __repr__(self):
        return "Rect(%s, %s)" % (self.tl, self.br)


def neighbours(x, y, grid):
    """Return the coordinates and value of all neighbours of (x, y)"""
    n = set()
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        with suppress(KeyError):
            n.add((x + dx, y + dy, grid[(x + dx, y + dy)]))
    return n


def nibble_to_bin(hexchar):
    return bin(int(hexchar, 16))[2:].zfill(4)


def lmap(op, array):
    return list(map(op, array))


def sign(n: int) -> int:
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0


def unique(lst):
    return set(lst)


def manhattan_distance(p0, p1) -> int:
    return sum(abs(a - b) for a, b in zip(p0, p1))


def unpack(data: str, sep="\n", fn=str):
    sections = data.strip().split(sep)
    return [fn(section) for section in sections]


def slurp(filename: str) -> str:
    with open(filename) as f:
        return f.read().rstrip()


def xor(a, b):
    return bool(a) ^ bool(b)


def assert_eq(a, b):
    assert a == b, "%s == %s" % (a, b)


def assert_ge(a, b):
    assert a >= b, "%s >= %s" % (a, b)


def chunks(data, size):
    """split a list into a list of lists"""
    for i in range(0, len(data), size):
        yield data[i : i + size]


def take(data, items):
    for _ in range(items):
        yield data.pop(0)


def numbers(s):
    """return all numbers in a string"""
    return lmap(int, re.findall(r"-?\d+", s))


def digits(s):
    """return list of digits in a string"""
    return lmap(int, re.findall(r"\d", s))


def revstr(s):
    """reverse a string"""
    return s[::-1]


def shortest_path(start, end, edges):
    """
    Get the path with lowest weight from start to end

    input:
        edges: edges[from][to] = weight

        edges = defaultdict(lambda: defaultdict(int))
    return:
        list of tuples ((x, y), totalweight) excluding start in reverse order
    """
    paths = {start: (None, 0)}
    Q = PriorityQueue()
    visited = set()
    current = start
    while current != end:
        visited.add(current)
        current_weight = paths[current][1]
        for node, weight in edges[current].items():
            total = current_weight + weight
            if node not in paths:
                # first visit
                paths[node] = (current, total)
            elif total < paths[node][1]:
                # better path
                paths[node] = (current, total)
            else:
                # do nothing
                pass
            Q.put((total, node))
        try:
            _, next_ = Q.get(block=False)
            while next_ in visited:
                _, next_ = Q.get(block=False)
        except Empty:
            return []
        current = next_
    current = end
    path = []
    while current != start:
        path.append(paths[current])
        current = paths[current][0]
    return path


def transpose(lol):
    """
    Transpose list of list:
        [[1,2,3,4], [5,6,7,8]] -> [[1,5], [2,6], [3,7], [4,8]]
    """
    t = [[0 for _, _ in enumerate(lol)] for _, _ in enumerate(lol[0])]
    for x, row in enumerate(lol):
        for y, val in enumerate(row):
            t[y][x] = val
    return t


def puzzlenum(file):
    return os.path.splitext(os.path.basename(file))[0]


def puzzlefile(file):
    return f"{puzzlenum(file)}.txt"


def parse_grid(s, fn=str, ignore=""):
    """
    Generates a dict of the grid with coordinates as keys:
        G[(x, y)] = val

    - `fn` applies the function `fn` on val
    - `ignore` contains a list of characters to ignore

    (0, 0) ------> (0, 9)
      |              |
      |              |
      |              |
      |              |
      v              v
    (9, 0) ------> (9, 9)

    Returns tuple: (G, xmax, ymax)
    """
    grid = {}
    for y, row in enumerate(unpack(s)):
        for x, c in enumerate(list(row)):
            if c not in ignore:
                grid[(x, y)] = fn(c)
        xmax = x
    ymax = y
    return grid, xmax, ymax


def line(x, y, heading, len_):
    """
    Generate coordinates for a line starting at (x, y) with specified heading
    and length.
    """
    HEADINGS = {
        "E": (1, 0),
        "NE": (1, 1),
        "N": (0, 1),
        "NW": (-1, 1),
        "W": (-1, 0),
        "SW": (-1, -1),
        "S": (0, -1),
        "SE": (1, -1),
    }
    dx, dy = HEADINGS[heading]
    for i in range(len_):
        yield x + i * dx, y + i * dy


def cat(things, sep=""):
    """Concatenate the things."""
    return sep.join(map(str, things))


def middle(lst):
    return lst[len(lst) // 2]


def print_grid(G, xlim=None, ylim=None, empty="."):
    """
    Print a grid on the console

    Posibility to specify the x and y limits, otherwise xlim/ylim is extracted
    from the grid.
    """
    if xlim:
        xmin = xlim[0]
        xmax = xlim[1]
    else:
        xmin = min([x for x, _ in G.keys()])
        xmax = max([x for x, _ in G.keys()])
    if ylim:
        ymin = ylim[0]
        ymax = ylim[1]
    else:
        ymin = min([y for _, y in G.keys()])
        ymax = max([y for _, y in G.keys()])

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            try:
                print(G[(x, y)], end="")
            except KeyError:
                print(empty, end="")
        print()


def timer(func):
    def wrapper(*args, **kwargs):
        tic = time.perf_counter()  # monotonic()
        result = func(*args, **kwargs)
        toc = time.perf_counter()  # monotonic()
        ms = round(1000 * (toc - tic), 3)
        print(f"[{func.__name__!s}: {ms}ms]", end=" ")
        return result

    return wrapper


if __name__ == "__main__":
    # chunks
    assert_eq(["ab", "cd", "ef", "gh"], list(chunks("abcdefgh", 2)))
    assert_eq(["abcd", "efgh"], list(chunks("abcdefgh", 4)))

    # take
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert_eq([1, 2, 3], list(take(data, 3)))
    assert_eq([4, 5, 6, 7, 8, 9], list(take(data, 6)))
    assert_eq([], data)

    # numbers
    assert_eq([1, 2, 3, 4], numbers("hell1o world 2=3++++++4"))

    # rect
    r = Rect(P2d(1, 1), P2d(3, 3))
    assert r.overlap(Rect(P2d(1, 1), P2d(2, 2)))
    assert not r.overlap(Rect(P2d(4, 4), P2d(5, 5)))
    assert r.overlap(Rect(P2d(0, 1), P2d(1, 1)))

    assert_eq([[1, 5], [2, 6], [3, 7], [4, 8]], transpose([[1, 2, 3, 4], [5, 6, 7, 8]]))
