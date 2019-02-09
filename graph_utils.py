#---------------------------------------------
#
#  Methods to draw graphs and contours with matplotlib
#
#  J.A. Hernando
#  February 2019
#---------------------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.style.use('ggplot')

figsize = 5, 3.8
figsize = 7, 4.4
cmap    = 'hot'

def fun1d(fun, x0 = 0., length = 3., newfig = True):
    xs   = np.linspace(x0 - length, x0 + length, 100)
    fig = plt.figure() if newfig else plt.gcf(); ax = plt.gca()
    ax.plot(xs, fun(xs))
    ax.set_xlabel('$x$'); ax.set_ylabel('f(x)')
    return fig, ax


def graph(fun, x0 = 0., y0 = 0., length = 3., zlim = None, newfig = True):
    """ draws a funcion
    parameters:
        fun    : (function) x,y function to draw
        x0, y0 : (float)    values of the center of the function
        length : (float)    length of the x, y axis
        zlim   : (float, float) zlimits
        newfig : (bool)     False is use existing figure
    returns:
    """
    xs  = np.linspace(x0 - length, x0 + length, 100)
    ys  = np.linspace(y0 - length, y0 + length, 100)
    xms, yms = np.meshgrid(xs, ys)
    zms = fun(xms, yms)
    if (zlim is not None):
        zmin, zmax = zlim
        zms[ zms < zmin] = zmin
        zms[ zms > zmax] = zmax
    fig = plt.figure(figsize=figsize) if newfig else plt.gcf()
    ax = plt.gca(projection='3d')
    sf  = ax.plot_surface(xms, yms, zms, cmap=cmap, alpha = 0.8)
    if (zlim is not None): ax.set_zlim3d(*zlim)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_aspect('equal')
    if (newfig): fig.colorbar(sf)
    return fig, ax


def contour(fun, contours=20, length = 3., zlim = None, fill = True, newfig = True):
    xs  = np.linspace(-1.*length, length, 100)
    ys  = np.linspace(-1.*length, length, 100)
    xms, yms = np.meshgrid(xs, ys)
    zms = fun(xms, yms)
    if (zlim is not None):
        zmin, zmax = zlim
        zms[ zms < zmin] = zmin
        zms[ zms > zmax] = zmax
    fig = plt.figure(figsize=figsize) if newfig else plt.gcf()
    ax = plt.gca()
    if (fill):
        c0 = ax.contourf(xms, yms, zms, contours, alpha=0.8, cmap=cmap)
    sf  = ax.contour(xms, yms, zms, contours, cmap=cmap)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_aspect('equal')
    fig.colorbar(sf)
    return fig, ax
    return


def arrow(x0, y0, vx, vy, head = 0.3):
    ax = plt.gca()
    vv = np.sqrt(vx*vx + vy*vy)
    if (vv == 0.):
        ax.plot(x0, y0, color='black', marker='*')
    else:
        vx, vy = (1-head)*vx, (1-head)*vy
        ax.arrow(x0, y0, vx, vy, head_width=head, head_length=head, fc='k',
                color='black', lw=2)
    return ax

def graph_section(fun, x0, y0, vx, vy, length = 3., sign = +1.):
    xs  = np.linspace(-1.*length, length, 100)
    ys  = np.linspace(-1.*length, length, 100)
    xms, yms = np.meshgrid(xs, ys)
    zms = fun(xms, yms)
    dvx, dvy = xms - x0, yms - y0
    dd  = dvy * vx - dvx * vy
    zmin = np.min(zms)
    zms[sign*dd >= 0.] = zmin
    fig = plt.figure(figsize=(8, 3.8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    sf = ax.plot_surface(xms, yms, zms, cmap=cmap)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_aspect('equal')
    fig.colorbar(sf)
    ax = plt.subplot(1, 2, 2)
    ts  = np.linspace(-1.*length, length, 100)
    gxs = x0 + ts * vx
    gys = y0 + ts * vy
    ffs = fun(gxs, gys)
    ax.plot(ts, ffs)
    sel = (ts >= 0.) & (ts < 0.1)
    ax.plot(ts[sel], ffs[sel], color = 'black', lw = 2)
    ax.set_xlabel('$t$'); ax.set_ylabel(r'$f({\bf x}_0 + t {\bf v})$');
    fig.tight_layout()
    return


def line2d(funx, funy, length = 1., newfig = True):
    ts = np.linspace(0., length, 100)
    xs, ys = funx(ts), funy(ts)
    fig = plt.figure() if newfig else plt.gcf()
    ax  = plt.gca()
    ax.plot(xs, ys)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_aspect('equal')
    return fig, ax

def line(funx, funy, funz, length = 1., newfig = True):
    ts = np.linspace(0., length, 100)
    xs, ys, zs = funx(ts), funy(ts), funz(ts)
    fig = plt.figure() if newfig else plt.gcf()
    ax  = plt.gca(projection='3d')
    ax.plot(xs, ys, zs)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_aspect('equal')
    return fig, ax
