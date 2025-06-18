import torch
import inspect
import functools
import operator


vol = lambda shape: functools.reduce(operator.mul, shape, 1)


class AffineMap:
    def __init__(self, fn):
        self.fn = fn
        self.dims = tuple(inspect.signature(fn).parameters)
        self.results = fn(*self.dims)
        self.num_dims = len(self.dims)

    def __call__(self, *args):
        return self.fn(*args)

    def __repr__(self):
        return f"({', '.join(self.dims)}) -> ({', '.join(self.results)})"


class Stats:
    def __init__(self, indexing_maps, shape):
        self.indexing_maps = indexing_maps
        self.results_to_index = {
            affine_map.results: i for i, affine_map in enumerate(indexing_maps)
        }
        self.accesses = [
            torch.zeros(affine_map(*shape), dtype=torch.int)
            for affine_map in indexing_maps
        ]
        self.order = [
            torch.zeros(affine_map(*shape), dtype=torch.int)
            for affine_map in indexing_maps
        ]
        self.counter = 0

    def update(self, dims, global_index):
        i = self.results_to_index[dims]
        indexing_map = self.indexing_maps[i]
        index = indexing_map(*global_index)
        self.accesses[i][index] += 1
        self.order[i][index] = self.counter
        self.counter += 1

    @staticmethod
    def calc_order(o):
        assert len(o.shape) == 2
        right = o[0, 1] if o.shape[1] > 1 else -1
        down = o[1, 0] if o.shape[0] > 1 else -1
        if right < down:
            return "Row-major"
        if right > down:
            return "Col-major"
        return "Illegal order"

    def __repr__(self):
        accesses = "\n".join(map(str, self.accesses))
        orders = "\n".join(self.calc_order(o) + "\n" + str(o) for o in self.order)
        return (
            "Stats(\n\n# Accesses\n"
            + f"{accesses}"
            + "\n\n# Orders\n"
            + f"{orders}"
            + "\n)"
        )


class Generic:
    def __init__(self, *indexing_maps):
        self.indexing_maps = indexing_maps
        self.dims = indexing_maps[0].dims
        for indexing_map in self.indexing_maps:
            assert indexing_map.dims == self.dims
        self.threads = sorted(
            thread
            for thread in dir(self)
            if thread.startswith("datamovement") or thread.startswith("compute")
        )

    def yield_(self, *dims):
        if self.suspend or self.mailbox.get(dims) is not None:
            self.suspend = True
            return
        if self.log_threads:
            print(self.current_thread, "yield", dims)
        self.stats.update(dims, self.current_index)
        self.mailbox[dims] = 1
        self.epoch += 1

    def await_(self, *dims):
        if self.suspend or self.mailbox.get(dims) is None:
            self.suspend = True
            return
        if self.log_threads:
            print(self.current_thread, "await", dims)
        self.mailbox[dims] = None
        self.epoch += 1

    def iter_index(self, dim):
        if type(dim) == str:
            dim = self.dims.index(dim)
        return self.current_index[dim]

    def _get_index(self, i, shape):
        return tuple((i // vol(shape[j + 1 :]) % shape[j]) for j in range(len(shape)))

    def __call__(self, *shape):
        self.epoch = 0
        self.mailbox = {}
        self.suspend = False
        self.current_thread = "unknown"
        self.current_index = None
        self.log_threads = False
        self.stats = Stats(self.indexing_maps, shape)

        for i in range(vol(shape)):
            staring_epoch = self.epoch
            self.current_index = self._get_index(i, shape)
            for thread in self.threads:
                self.suspend = False
                self.current_thread = thread
                getattr(self, thread)()
            assert self.epoch > staring_epoch, "we have a deadlock"

        return self.stats

    def __repr__(self):
        class_name = self.__class__.__name__
        indexing_maps = ",\n  ".join(map(str, self.indexing_maps))
        return f"{class_name}" + "(\n  " + f"{indexing_maps}" + "\n)"


class MatmulKMN(Generic):
    def __init__(self):
        super().__init__(
            AffineMap(lambda k, m, n: (m, k)),
            AffineMap(lambda k, m, n: (k, n)),
            AffineMap(lambda k, m, n: (m, n)),
        )

    def datamovement0(self):
        if self.iter_index("n") == 0:
            self.yield_("m", "k")

    def datamovement1(self):
        self.yield_("k", "n")

    def datamovement2(self):
        self.await_("m", "n")

    def compute(self):
        if self.iter_index("n") == 0:
            self.await_("m", "k")
        self.await_("k", "n")
        self.yield_("m", "n")


kmn = MatmulKMN()
print(kmn)
kmn_stats = kmn(2, 2, 4)
print(kmn_stats)
