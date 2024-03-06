import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from interpn import MultilinearRectilinear, MultilinearRegular, MulticubicRegular

if __name__ == "__main__":
    # 1D comparison
    xdata = np.arange(-2.0, 2.5, 0.5)
    ydata = xdata ** 2

    xinterp = np.arange(-3.0, 3.05, 0.05)

    dims = np.asarray([xdata.size])
    starts = np.asarray([-2.0])
    steps = np.asarray([0.5])
    y_interpn = MulticubicRegular.new(dims, starts, steps, ydata).eval([xinterp])

    y_sp = RegularGridInterpolator(
        [xdata], ydata, bounds_error=None, method="cubic"
    )(xinterp)

    plt.figure(figsize=(12, 8))
    plt.scatter(xdata, ydata, marker='o', color='k', s=20, label="Data")
    plt.plot(xinterp, y_interpn, color='k', linewidth=2, linestyle="-", label="Interpn MulticubicRegular")
    plt.plot(xinterp, y_sp, color='k', linewidth=3, linestyle="dotted", label="Scipy RegularGridInterpolator Cubic")
    plt.legend()

    plt.figure(figsize=(12, 8))
    inds = np.where(np.where(xinterp >= -2.05, True, False) * np.where(xinterp <= 2.0, True, False))
    # plt.scatter(xdata, ydata, marker='o', color='k', s=20, label="Data")
    plt.plot(xinterp[inds], (y_interpn - xinterp**2)[inds], color='k', linewidth=2, linestyle="-", label="Error, Interpn MulticubicRegular")
    plt.plot(xinterp[inds], (y_sp - xinterp**2)[inds], color='k', linewidth=3, linestyle="dotted", label="Error, Scipy RegularGridInterpolator Cubic")
    plt.legend()

    # 2D comparison
    xdata = np.linspace(-3.0, 3.0, 6, endpoint=True)
    ydata = np.linspace(-3.0, 3.0, 6, endpoint=True)
    xmesh, ymesh = np.meshgrid(xdata, ydata, indexing="ij")
    zmesh = xmesh**2 + ymesh**2

    dims = np.asarray([6, 6])
    starts = np.asarray([-3.0, -3.0])
    steps = np.asarray([xmesh[1, 0] - xmesh[0,0], ymesh[0, 1] - ymesh[0, 0]])

    xinterp = np.linspace(-5.0, 5.0, 30, endpoint=True)
    yinterp = np.linspace(-5.0, 5.0, 30, endpoint=True)
    xinterpmesh, yinterpmesh = np.meshgrid(xinterp, yinterp, indexing="ij")
    zinterp = xinterpmesh**2 + yinterpmesh**2

    z_interpn = MulticubicRegular.new(dims, starts, steps, zmesh).eval([xinterpmesh.flatten(), yinterpmesh.flatten()]).reshape(xinterpmesh.shape)

    z_sp = RegularGridInterpolator(
        [xdata, ydata], zmesh, bounds_error=None, fill_value=None, method="cubic"
    )((xinterpmesh, yinterpmesh))

    _fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    plt.suptitle("Quadratic Test Function")
    for i, (z, title) in enumerate([(zinterp, "Truth"), (z_interpn, "Interpn MulticubicRegular"), (z_sp, "Scipy RegularGridInterp. Cubic")]):
        plt.sca(axes[i])
        plt.imshow(z.T, origin="lower", extent=[-5.0, 5.0, -5.0, 5.0])
        plt.contour(xinterpmesh, yinterpmesh, z.T, colors='k', levels=6)
        plt.gca().add_patch(plt.Rectangle((-3.0, -3.0), 6.0, 6.0, edgecolor='w', fill=False, label="Interpolating Region"))
        plt.title(title)
        plt.legend()

        plt.sca(axes[i + 3])
        plt.imshow((z - zinterp).T, origin="lower", extent=[-5.0, 5.0, -5.0, 5.0])
        plt.gca().add_patch(plt.Rectangle((-3.0, -3.0), 6.0, 6.0, edgecolor='w', fill=False, label="Interpolating Region"))
        plt.title("Error\n" + title)
        plt.axis("off")
        plt.colorbar()
        plt.legend()

    testing = (len(sys.argv) > 1) and (sys.argv[1] == "test")
    if not testing:
        plt.show()
