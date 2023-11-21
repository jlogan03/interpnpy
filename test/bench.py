import gc

from timeit import timeit
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from interpn import MultilinearRectilinear, MultilinearRegular


def bench_8_dims_1_obs():
    nbench = 100  # Bench iterations
    preallocate = False  # Whether to preallocate output array for interpn
    ndims = 8  # Number of grid dimensions
    ngrid = 2  # Size of grid on each dimension
    nobs = int(1)  # Number of observation points
    m = max(int(float(nobs) ** (1.0 / ndims) + 2), 2)

    grids = [np.linspace(-1.0, 1.0, ngrid) for _ in range(ndims)]
    xgrid = np.meshgrid(*grids, indexing="ij")
    zgrid = np.random.uniform(-1.0, 1.0, xgrid[0].size)

    dims = [x.size for x in grids]
    starts = np.array([x[0] for x in grids])
    steps = np.array([x[1] - x[0] for x in grids])

    # Baseline interpolating on the same domain,
    # keeping the points entirely inside the domain to give a clear
    # cut between interpolation and extrapolation
    obsgrid = np.meshgrid(
        *[np.linspace(-0.99, 0.99, m) for _ in range(ndims)], indexing="ij"
    )
    obsgrid = [x.flatten()[0:nobs] for x in obsgrid]  # Trim to the exact right number

    # Initialize all interpolator methods
    # Scipy RegularGridInterpolator is actually a more general rectilinear method
    rectilinear_sp = RegularGridInterpolator(
        grids, zgrid.reshape(xgrid[0].shape), bounds_error=None
    )
    rectilinear_interpn = MultilinearRectilinear.new(grids, zgrid)
    regular_interpn = MultilinearRegular.new(dims, starts, steps, zgrid)

    # Preallocate output for potential perf advantage
    # Allocate at eval for 1:1 comparison with scipy
    out = None if not preallocate else np.zeros_like(obsgrid[0].flatten())
    interps = {
        "scipy RegularGridInterpolator": rectilinear_sp,
        "interpn MultilinearRegular": lambda p: regular_interpn.eval(p, out),
        "interpn MultilinearRectilinear": lambda p: rectilinear_interpn.eval(p, out),
        "numpy interp": lambda p: np.interp(p[0], grids[0], zgrid),  # 1D only
    }

    # Interpolation in sequential order
    points_interpn = [x.flatten() for x in obsgrid]
    points_sp = np.array(points_interpn).T
    points = {
        "scipy RegularGridInterpolator": points_sp,
        "interpn MultilinearRegular": points_interpn,
        "interpn MultilinearRectilinear": points_interpn,
        "numpy interp": points_interpn,
    }

    print("\nInterpolation in sequential order")
    for name, func in interps.items():
        if name == "numpy interp" and ndims > 1:
            continue
        p = points[name]
        timeit(lambda: func(p), number=nbench)  # warmup
        t = timeit(lambda: func(p), number=nbench) / nbench
        throughput = nobs / t
        print("----")
        print(f"Method: {name}")
        print(f"Time {t:.2e} s")
        print(f"Throughput {throughput:.2e} #/s")

    # Interpolation in random order
    points_interpn = [np.random.permutation(x.flatten()) for x in obsgrid]
    points_sp = np.array(points_interpn).T
    points = {
        "scipy RegularGridInterpolator": points_sp,
        "interpn MultilinearRegular": points_interpn,
        "interpn MultilinearRectilinear": points_interpn,
        "numpy interp": points_interpn,
    }

    print("\nInterpolation in random order")
    for name, func in interps.items():
        if name == "numpy interp" and ndims > 1:
            continue
        p = points[name]
        timeit(lambda: func(p), number=nbench)  # warmup
        t = timeit(lambda: func(p), number=nbench) / nbench
        throughput = nobs / t
        print("----")
        print(f"Method: {name}")
        print(f"Time {t:.2e} s")
        print(f"Throughput {throughput:.2e} #/s")

    # Extrapolation in corner region in random order
    points_interpn = [np.random.permutation(x.flatten()) + 3.0 for x in obsgrid]
    points_sp = np.array(points_interpn).T
    points = {
        "scipy RegularGridInterpolator": points_sp,
        "interpn MultilinearRegular": points_interpn,
        "interpn MultilinearRectilinear": points_interpn,
        "numpy interp": points_interpn,
    }

    print("\nExtrapolation to corner region in random order")
    for name, func in interps.items():
        if name == "numpy interp" and ndims > 1:
            continue
        p = points[name]
        timeit(lambda: func(p), number=nbench)  # warmup
        t = timeit(lambda: func(p), number=nbench) / nbench
        throughput = nobs / t
        print("----")
        print(f"Method: {name}")
        print(f"Time {t:.2e} s")
        print(f"Throughput {throughput:.2e} #/s")

    # Extrapolation in side region in random order
    points_interpn = [
        np.random.permutation(x.flatten()) + (3.0 if i == 0 else 0.0)
        for i, x in enumerate(obsgrid)
    ]
    points_sp = np.array(points_interpn).T
    points = {
        "scipy RegularGridInterpolator": points_sp,
        "interpn MultilinearRegular": points_interpn,
        "interpn MultilinearRectilinear": points_interpn,
        "numpy interp": points_interpn,
    }

    print("\nExtrapolation to side region in random order")
    for name, func in interps.items():
        if name == "numpy interp" and ndims > 1:
            continue
        p = points[name]
        t = timeit(lambda: func(p), number=nbench) / nbench
        throughput = nobs / t
        print("----")
        print(f"Method: {name}")
        print(f"Time {t:.2e} s")
        print(f"Throughput {throughput:.2e} #/s")


def bench_3_dims_n_obs_unordered():
    nbench = 1000  # Bench iterations

    for preallocate in [False, True]:
        ndims = 3  # Number of grid dimensions
        ngrid = 20  # Size of grid on each dimension

        grids = [np.linspace(-1.0, 1.0, ngrid) for _ in range(ndims)]
        xgrid = np.meshgrid(*grids, indexing="ij")
        zgrid = np.random.uniform(-1.0, 1.0, xgrid[0].size)

        dims = [x.size for x in grids]
        starts = np.array([x[0] for x in grids])
        steps = np.array([x[1] - x[0] for x in grids])

        # Initialize all interpolator methods
        # Scipy RegularGridInterpolator is actually a more general rectilinear method
        rectilinear_sp = RegularGridInterpolator(
            grids, zgrid.reshape(xgrid[0].shape), bounds_error=None
        )
        rectilinear_interpn = MultilinearRectilinear.new(grids, zgrid)
        regular_interpn = MultilinearRegular.new(dims, starts, steps, zgrid)

        throughputs = {
            "scipy RegularGridInterpolator": [],
            "interpn MultilinearRegular": [],
            "interpn MultilinearRectilinear": [],
        }
        ns = np.logspace(0, 5, 40, base=10)
        ns = [int(x) for x in ns]
        # ns = [1, 10, 100, 1000, 10000, 50000, 100000]
        print("\nThroughput plotting")
        print(ns)
        for nobs in ns:
            print(nobs)
            m = max(int(float(nobs) ** (1.0 / ndims) + 2), 2)

            # Baseline interpolating on the same domain,
            # keeping the points entirely inside the domain to give a clear
            # cut between interpolation and extrapolation
            obsgrid = np.meshgrid(
                *[np.linspace(-0.99, 0.99, m) for _ in range(ndims)], indexing="ij"
            )
            obsgrid = [
                x.flatten()[0:nobs] for x in obsgrid
            ]  # Trim to the exact right number

            # Preallocate output for potential perf advantage
            # Allocate at eval for 1:1 comparison with scipy
            out = None if not preallocate else np.zeros_like(obsgrid[0].flatten())
            interps = {
                "scipy RegularGridInterpolator": rectilinear_sp,
                "interpn MultilinearRegular": lambda p: regular_interpn.eval(p, out),
                "interpn MultilinearRectilinear": lambda p: rectilinear_interpn.eval(
                    p, out
                ),
            }

            # Interpolation in random order
            points_interpn = [np.random.permutation(x.flatten()) for x in obsgrid]
            points_sp = np.array(points_interpn).T
            points = {
                "scipy RegularGridInterpolator": points_sp,
                "interpn MultilinearRegular": points_interpn,
                "interpn MultilinearRectilinear": points_interpn,
            }

            for name, func in interps.items():
                p = points[name]
                timeit(
                    lambda: func(p), setup=gc.collect, number=int(nbench / 4)
                )  # warmup
                t = timeit(lambda: func(p), setup=gc.collect, number=nbench) / nbench
                throughput = nobs / t
                throughputs[name].append(throughput)

        linestyles = ["dotted", "-", "--"]
        alpha = [0.5, 1.0, 1.0]
        plt.figure(figsize=(12, 8))
        all_throughputs = sum([v for v in throughputs.values()], [])
        max_throughput = max(all_throughputs)
        for i, (k, v) in enumerate(throughputs.items()):
            normalized_throughput = np.array(v) / max_throughput
            plt.loglog(
                ns,
                normalized_throughput,
                color="k",
                linewidth=2,
                linestyle=linestyles[i],
                label=k,
                alpha=alpha[i],
            )
        plt.legend()
        plt.xlabel("Number of Observation Points")
        plt.ylabel("Normalized Throughput [1/s]")
        with_alloc_string = (
            "\nWith Preallocated Output"
            if preallocate
            else "\nWithout Preallocated Output"
        )
        plt.title("Interpolation on 20x20x20 Grid" + with_alloc_string)
        plt.show()


def bench_8_dims_n_obs_unordered():
    nbench = 100  # Bench iterations

    for preallocate in [False, True]:
        ndims = 8  # Number of grid dimensions
        ngrid = 2  # Size of grid on each dimension

        grids = [np.linspace(-1.0, 1.0, ngrid) for _ in range(ndims)]
        xgrid = np.meshgrid(*grids, indexing="ij")
        zgrid = np.random.uniform(-1.0, 1.0, xgrid[0].size)

        dims = [x.size for x in grids]
        starts = np.array([x[0] for x in grids])
        steps = np.array([x[1] - x[0] for x in grids])

        # Initialize all interpolator methods
        # Scipy RegularGridInterpolator is actually a more general rectilinear method
        rectilinear_sp = RegularGridInterpolator(
            grids, zgrid.reshape(xgrid[0].shape), bounds_error=None
        )
        rectilinear_interpn = MultilinearRectilinear.new(grids, zgrid)
        regular_interpn = MultilinearRegular.new(dims, starts, steps, zgrid)

        throughputs = {
            "scipy RegularGridInterpolator": [],
            "interpn MultilinearRegular": [],
            "interpn MultilinearRectilinear": [],
        }
        # ns = np.logspace(0, 4, 40, base=10)
        # ns = [int(x) for x in ns]
        ns = [1, 10, 50, 100, 500, 1000, 10000, 100000]
        print("\nThroughput plotting")
        print(ns)
        for nobs in ns:
            print(nobs)
            m = max(int(float(nobs) ** (1.0 / ndims) + 2), 2)

            # Baseline interpolating on the same domain,
            # keeping the points entirely inside the domain to give a clear
            # cut between interpolation and extrapolation
            obsgrid = np.meshgrid(
                *[np.linspace(-0.99, 0.99, m) for _ in range(ndims)], indexing="ij"
            )
            obsgrid = [
                x.flatten()[0:nobs] for x in obsgrid
            ]  # Trim to the exact right number

            # Preallocate output for potential perf advantage
            # Allocate at eval for 1:1 comparison with scipy
            out = None if not preallocate else np.zeros_like(obsgrid[0].flatten())
            interps = {
                "scipy RegularGridInterpolator": rectilinear_sp,
                "interpn MultilinearRegular": lambda p: regular_interpn.eval(p, out),
                "interpn MultilinearRectilinear": lambda p: rectilinear_interpn.eval(
                    p, out
                ),
            }

            # Interpolation in random order
            points_interpn = [np.random.permutation(x.flatten()) for x in obsgrid]
            points_sp = np.array(points_interpn).T
            points = {
                "scipy RegularGridInterpolator": points_sp,
                "interpn MultilinearRegular": points_interpn,
                "interpn MultilinearRectilinear": points_interpn,
            }

            for name, func in interps.items():
                p = points[name]
                timeit(
                    lambda: func(p), setup=gc.collect, number=int(nbench / 4)
                )  # warmup
                t = timeit(lambda: func(p), setup=gc.collect, number=nbench) / nbench
                throughput = nobs / t
                throughputs[name].append(throughput)

        linestyles = ["dotted", "-", "--"]
        alpha = [0.5, 1.0, 1.0]
        plt.figure(figsize=(12, 8))
        all_throughputs = sum([v for v in throughputs.values()], [])
        max_throughput = max(all_throughputs)
        for i, (k, v) in enumerate(throughputs.items()):
            normalized_throughput = np.array(v) / max_throughput
            plt.loglog(
                ns,
                normalized_throughput,
                color="k",
                linewidth=2,
                linestyle=linestyles[i],
                label=k,
                alpha=alpha[i],
            )
        plt.legend()
        plt.xlabel("Number of Observation Points")
        plt.ylabel("Normalized Throughput [1/s]")
        with_alloc_string = (
            "\nWith Preallocated Output"
            if preallocate
            else "\nWithout Preallocated Output"
        )
        plt.title("Interpolation on 2x...x2 8D Grid" + with_alloc_string)
        plt.show()


if __name__ == "__main__":
    bench_8_dims_1_obs()
    bench_8_dims_n_obs_unordered()
    bench_3_dims_n_obs_unordered()
