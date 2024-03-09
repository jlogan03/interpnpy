import gc

from timeit import timeit
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from interpn import MultilinearRectilinear, MultilinearRegular, MulticubicRegular


def bench_6_dims_1_obs():
    nbench = 100  # Bench iterations
    preallocate = False  # Whether to preallocate output array for InterpN
    ndims = 6  # Number of grid dimensions
    ngrid = 4  # Size of grid on each dimension
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
    cubic_rectilinear_sp = RegularGridInterpolator(
        grids, zgrid.reshape(xgrid[0].shape), bounds_error=None, method="cubic"
    )
    rectilinear_interpn = MultilinearRectilinear.new(grids, zgrid)
    regular_interpn = MultilinearRegular.new(dims, starts, steps, zgrid)
    cubic_regular_interpn = MulticubicRegular.new(dims, starts, steps, zgrid)

    # Preallocate output for potential perf advantage
    # Allocate at eval for 1:1 comparison with Scipy
    out = None if not preallocate else np.zeros_like(obsgrid[0].flatten())
    interps = {
        "Scipy RegularGridInterpolator Linear": rectilinear_sp,
        "Scipy RegularGridInterpolator Cubic": cubic_rectilinear_sp,
        "InterpN MultilinearRegular": lambda p: regular_interpn.eval(p, out),
        "InterpN MultilinearRectilinear": lambda p: rectilinear_interpn.eval(p, out),
        "InterpN MulticubicRegular": lambda p: cubic_regular_interpn.eval(p, out),
        "numpy interp": lambda p: np.interp(p[0], grids[0], zgrid),  # 1D only
    }

    # Interpolation in sequential order
    points_interpn = [x.flatten() for x in obsgrid]
    points_sp = np.array(points_interpn).T
    points = {
        "Scipy RegularGridInterpolator Linear": points_sp,
        "Scipy RegularGridInterpolator Cubic": points_sp,
        "InterpN MultilinearRegular": points_interpn,
        "InterpN MultilinearRectilinear": points_interpn,
        "InterpN MulticubicRegular": points_interpn,
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
        "Scipy RegularGridInterpolator Linear": points_sp,
        "Scipy RegularGridInterpolator Cubic": points_sp,
        "InterpN MultilinearRegular": points_interpn,
        "InterpN MultilinearRectilinear": points_interpn,
        "InterpN MulticubicRegular": points_interpn,
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
        "Scipy RegularGridInterpolator Linear": points_sp,
        "Scipy RegularGridInterpolator Cubic": points_sp,
        "InterpN MultilinearRegular": points_interpn,
        "InterpN MultilinearRectilinear": points_interpn,
        "InterpN MulticubicRegular": points_interpn,
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
        "Scipy RegularGridInterpolator Linear": points_sp,
        "Scipy RegularGridInterpolator Cubic": points_sp,
        "InterpN MultilinearRegular": points_interpn,
        "InterpN MultilinearRectilinear": points_interpn,
        "InterpN MulticubicRegular": points_interpn,
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
    nbench = 100  # Bench iterations

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
        cubic_rectilinear_sp = RegularGridInterpolator(
            grids, zgrid.reshape(xgrid[0].shape), bounds_error=None, method="cubic"
        )
        rectilinear_interpn = MultilinearRectilinear.new(grids, zgrid)
        regular_interpn = MultilinearRegular.new(dims, starts, steps, zgrid)
        cubic_regular_interpn = MulticubicRegular.new(dims, starts, steps, zgrid)

        throughputs = {
            "Scipy RegularGridInterpolator Linear": [],
            "Scipy RegularGridInterpolator Cubic": [],
            "InterpN MultilinearRegular": [],
            "InterpN MultilinearRectilinear": [],
            "InterpN MulticubicRegular": [],
        }
        # ns = np.logspace(0, 5, 10, base=10)
        # ns = [int(x) for x in ns]
        # ns = sorted(list(set(ns)))
        ns = [1, 10, 50, 100, 500, 1000, 10000, 100000]
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
            # Allocate at eval for 1:1 comparison with Scipy
            out = None if not preallocate else np.zeros_like(obsgrid[0].flatten())
            interps = {
                "Scipy RegularGridInterpolator Linear": rectilinear_sp,
                "Scipy RegularGridInterpolator Cubic": cubic_rectilinear_sp,
                "InterpN MultilinearRegular": lambda p: regular_interpn.eval(p, out),
                "InterpN MultilinearRectilinear": lambda p: rectilinear_interpn.eval(
                    p, out
                ),
                "InterpN MulticubicRegular": lambda p: cubic_regular_interpn.eval(
                    p, out
                ),
            }

            # Interpolation in random order
            points_interpn = [np.random.permutation(x.flatten()) for x in obsgrid]
            points_sp = np.array(points_interpn).T
            points = {
                "Scipy RegularGridInterpolator Linear": points_sp,
                "Scipy RegularGridInterpolator Cubic": points_sp,
                "InterpN MultilinearRegular": points_interpn,
                "InterpN MultilinearRectilinear": points_interpn,
                "InterpN MulticubicRegular": points_interpn,
            }

            for name, func in interps.items():
                if "cubic" in name.lower() and nobs > 10000:
                    continue
                p = points[name]
                timeit(
                    lambda: func(p), setup=gc.collect, number=int(nbench / 4)
                )  # warmup
                t = timeit(lambda: func(p), setup=gc.collect, number=nbench) / nbench
                throughput = nobs / t
                throughputs[name].append(throughput)

        linestyles = ["dotted", "-", "--", "-.", (0, (3, 1, 1, 1, 1, 1))]
        alpha = [0.5, 1.0, 1.0, 1.0, 1.0]

        _fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plt.suptitle("Interpolation on 20x20x20 Grid")
        for i, kind in enumerate(["Linear", "Cubic"]):
            # plt.figure()
            plt.sca(axes[i])
            throughputs_this_kind = [
                (k, v) for k, v in throughputs.items() if kind.lower() in k.lower()
            ]
            all_throughputs_this_kind = sum([v for _, v in throughputs_this_kind], [])
            max_throughput = max(all_throughputs_this_kind)
            for i, (k, v) in enumerate(throughputs_this_kind):
                normalized_throughput = np.array(v) / max_throughput
                plt.loglog(
                    ns[: normalized_throughput.size],
                    normalized_throughput,
                    color="k",
                    linewidth=2,
                    linestyle=linestyles[i],
                    label=k,
                    alpha=alpha[i],
                )
            plt.legend()
            plt.xlabel("Number of Observation Points")
            plt.ylabel("Normalized Throughput")
            with_alloc_string = "\nWith Preallocated Output" if preallocate else ""
            plt.title(f"{kind}" + with_alloc_string)
        plt.show(block=False)


def bench_6_dims_n_obs_unordered():
    nbench = 10  # Bench iterations

    for preallocate in [False, True]:
        ndims = 6  # Number of grid dimensions
        ngrid = 4  # Size of grid on each dimension

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
        cubic_rectilinear_sp = RegularGridInterpolator(
            grids, zgrid.reshape(xgrid[0].shape), bounds_error=None, method="cubic"
        )
        rectilinear_interpn = MultilinearRectilinear.new(grids, zgrid)
        regular_interpn = MultilinearRegular.new(dims, starts, steps, zgrid)
        cubic_regular_interpn = MulticubicRegular.new(dims, starts, steps, zgrid)

        throughputs = {
            "Scipy RegularGridInterpolator Linear": [],
            "Scipy RegularGridInterpolator Cubic": [],
            "InterpN MultilinearRegular": [],
            "InterpN MultilinearRectilinear": [],
            "InterpN MulticubicRegular": [],
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
            # Allocate at eval for 1:1 comparison with Scipy
            out = None if not preallocate else np.zeros_like(obsgrid[0].flatten())
            interps = {
                "Scipy RegularGridInterpolator Linear": rectilinear_sp,
                "Scipy RegularGridInterpolator Cubic": cubic_rectilinear_sp,
                "InterpN MultilinearRegular": lambda p: regular_interpn.eval(p, out),
                "InterpN MultilinearRectilinear": lambda p: rectilinear_interpn.eval(
                    p, out
                ),
                "InterpN MulticubicRegular": lambda p: cubic_regular_interpn.eval(
                    p, out
                ),
            }

            # Interpolation in random order
            points_interpn = [np.random.permutation(x.flatten()) for x in obsgrid]
            points_sp = np.array(points_interpn).T
            points = {
                "Scipy RegularGridInterpolator Linear": points_sp,
                "Scipy RegularGridInterpolator Cubic": points_sp,
                "InterpN MultilinearRegular": points_interpn,
                "InterpN MultilinearRectilinear": points_interpn,
                "InterpN MulticubicRegular": points_interpn,
            }

            for name, func in interps.items():
                p = points[name]
                timeit(
                    lambda: func(p), setup=gc.collect, number=int(nbench / 4)
                )  # warmup
                t = timeit(lambda: func(p), setup=gc.collect, number=nbench) / nbench
                throughput = nobs / t
                throughputs[name].append(throughput)

        linestyles = ["dotted", "-", "--", "-.", (0, (3, 1, 1, 1, 1, 1))]
        alpha = [0.5, 1.0, 1.0, 1.0, 1.0]
        _fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plt.suptitle("Interpolation on 4x...x4 6D Grid")
        for i, kind in enumerate(["Linear", "Cubic"]):
            # plt.figure()
            plt.sca(axes[i])
            throughputs_this_kind = [
                (k, v) for k, v in throughputs.items() if kind.lower() in k.lower()
            ]
            all_throughputs_this_kind = sum([v for _, v in throughputs_this_kind], [])
            max_throughput = max(all_throughputs_this_kind)
            for i, (k, v) in enumerate(throughputs_this_kind):
                normalized_throughput = np.array(v) / max_throughput
                plt.loglog(
                    ns[: normalized_throughput.size],
                    normalized_throughput,
                    color="k",
                    linewidth=2,
                    linestyle=linestyles[i],
                    label=k,
                    alpha=alpha[i],
                )
            plt.legend()
            plt.xlabel("Number of Observation Points")
            plt.ylabel("Normalized Throughput")
            with_alloc_string = "\nWith Preallocated Output" if preallocate else ""
            plt.title(f"{kind}" + with_alloc_string)
        plt.show(block=False)

def bench_throughput_vs_dims():
    nbench = 10
    throughputs = {
        "Scipy RegularGridInterpolator Linear": [],
        "Scipy RegularGridInterpolator Cubic": [],
        "InterpN MultilinearRegular": [],
        "InterpN MultilinearRectilinear": [],
        "InterpN MulticubicRegular": [],
    }
    ndims_to_test = [x for x in range(1, 9)]
    for ndims in ndims_to_test:
        nobs = 1000
        ngrid = 4  # Size of grid on each dimension

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
        cubic_rectilinear_sp = RegularGridInterpolator(
            grids, zgrid.reshape(xgrid[0].shape), bounds_error=None, method="cubic"
        )
        rectilinear_interpn = MultilinearRectilinear.new(grids, zgrid)
        regular_interpn = MultilinearRegular.new(dims, starts, steps, zgrid)
        cubic_regular_interpn = MulticubicRegular.new(dims, starts, steps, zgrid)

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
        # Allocate at eval for 1:1 comparison with Scipy
        interps = {
            "Scipy RegularGridInterpolator Linear": rectilinear_sp,
            "Scipy RegularGridInterpolator Cubic": cubic_rectilinear_sp,
            "InterpN MultilinearRegular": lambda p: regular_interpn.eval(p),
            "InterpN MultilinearRectilinear": lambda p: rectilinear_interpn.eval(
                p
            ),
            "InterpN MulticubicRegular":  lambda p: cubic_regular_interpn.eval(p),
        }

        # Interpolation in random order
        points_interpn = [np.random.permutation(x.flatten()) for x in obsgrid]
        points_sp = np.ascontiguousarray(np.array(points_interpn).T)
        points = {
            "Scipy RegularGridInterpolator Linear": points_sp,
            "Scipy RegularGridInterpolator Cubic": points_sp,
            "InterpN MultilinearRegular": points_interpn,
            "InterpN MultilinearRectilinear": points_interpn,
            "InterpN MulticubicRegular": points_interpn,
        }

        for name, func in interps.items():
            print(ndims, name)
            p = points[name]
            timeit(
                lambda: func(p), setup=gc.collect, number=int(nbench / 4)
            )  # warmup
            t = timeit(lambda: func(p), setup=gc.collect, number=nbench) / nbench
            throughput = nobs / t
            throughputs[name].append(throughput)

    linestyles = ["dotted", "-", "--", "-.", (0, (3, 1, 1, 1, 1, 1))]
    alpha = [0.5, 1.0, 1.0, 1.0, 1.0]

    _fig, axes = plt.subplots(1,2, figsize=(12,6), sharey=True)
    plt.suptitle(f"Interpolation on 4x...x4 N-Dimensional Grid\n{nobs} Observation Points")
    for i, kind in enumerate(["Linear", "Cubic"]):
        plt.sca(axes[i])
        throughputs_this_kind = [(k,v) for k, v in throughputs.items() if kind.lower() in k.lower()]
        all_throughputs_this_kind = sum([v for _, v in throughputs_this_kind], [])
        max_throughput = max(all_throughputs_this_kind)
        for i, (k, v) in enumerate(throughputs_this_kind):
            normalized_throughput = np.array(v) / max_throughput
            plt.semilogy(
                ndims_to_test,
                normalized_throughput,
                color="k",
                linewidth=2,
                linestyle=linestyles[i],
                label=k,
                alpha=alpha[i],
            )
        plt.legend()
        plt.xlabel("Number of Dimensions")
        plt.ylabel("Normalized Throughput")
        plt.title(kind)
        plt.show(block=False)

if __name__ == "__main__":
    bench_throughput_vs_dims()
    bench_6_dims_1_obs()
    bench_6_dims_n_obs_unordered()
    bench_3_dims_n_obs_unordered()
    plt.show(block=True)
