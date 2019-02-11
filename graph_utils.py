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
figsize = 8, 6
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
    fig = plt.figure(figsize = figsize) if newfig else plt.gcf()
    ax  = plt.gca()
    ax.plot(xs, ys)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_aspect('equal')
    return fig, ax

def line3d(trange, funx, funy, funz, color = 'blue', newfig = True):
    ts = np.linspace(*trange, 100)
    xs, ys, zs = funx(ts), funy(ts), funz(ts)
    fig = plt.figure(figsize = figsize) if newfig else plt.gcf()
    ax  = plt.gca(projection='3d')
    ax.plot(xs, ys, zs, color = color, lw = 2, alpha = 0.8)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_aspect('equal')
    return fig, ax

def wfsurface(urange, vrange, funx, funy, funz, nbins = 20, color = 'red', newfig = True):
    us = np.linspace(*urange, nbins)
    vs = np.linspace(*vrange, nbins)
    ums, vms = np.meshgrid(us, vs)
    xms, yms, zms = funx(ums, vms), funy(ums, vms), funz(ums, vms)
    fig = plt.figure(figsize = figsize) if newfig else plt.gcf()
    ax  = plt.gca(projection='3d')
    ax.plot_wireframe(xms, yms, zms, alpha = 0.8, color = color)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_aspect('equal')
    return fig, ax


def arrow3d(x0, y0, z0, vx, vy, vz, color = 'black', head = 0.3):
    ax = plt.gca(projection='3d')
    vv = np.sqrt(vx*vx + vy*vy + vz*vz)
    if (vv == 0.):
        ax.plot(x0, y0, z0, color='black', marker='*')
    else:
        vx, vy, vz = (1-head)*vx, (1-head)*vy, (1-head)*vz
        ax.quiver(x0, y0, z0, vx, vy, vz, color = color, lw=2)
    return ax


def sphere_plane(theta0,  phi0, length = 0.5):
    if (theta0 == np.pi/2.):
        print('Can not draw tangent plane at theta pi/2 ', theta0)
        return

    # draw sphere
    theta_range = 0., np.pi
    phi_range   = 0., 2.*np.pi
    r = 1.
    xpos = lambda theta, phi : r * np.cos(phi) * np.sin(theta)
    ypos = lambda theta, phi : r * np.sin(phi) * np.sin(theta)
    zpos = lambda theta, phi : r               * np.cos(theta)
    fig, ax = wfsurface(theta_range, phi_range, xpos, ypos, zpos);

    # draw plane
    fpx  = lambda x, y, z: 2*x
    fpy  = lambda x, y, z: 2*y
    fpz  = lambda x, y, z: 2*z
    x0, y0, z0 = xpos(theta0, phi0), ypos(theta0, phi0), zpos(theta0, phi0)
    fpx0, fpy0, fpz0 = fpx(x0, y0, z0), fpy(x0, y0, z0), fpz(x0, y0, z0)
    xplane = lambda x, y : x
    yplane = lambda x, y : y
    zplane = lambda x, y : z0 - (fpx0 * (x-x0) + fpy0 * (y - y0))/fpz0
    x_range = x0 - length, x0 + length
    y_range = y0 - length, y0 + length
    wfsurface(x_range, y_range, xplane, yplane, zplane, nbins =20, newfig = False)

    # draw gradient
    arrow3d(x0, y0, z0, fpx0, fpy0, fpz0)

    xptheta = lambda theta, phi :  r * np.cos(phi) * np.cos(theta)
    yptheta = lambda theta, phi :  r * np.sin(phi) * np.cos(theta)
    zptheta = lambda theta, phi : -r               * np.sin(theta)

    xpphi = lambda theta, phi : - r * np.sin(phi) * np.sin(theta)
    ypphi = lambda theta, phi :   r * np.cos(phi) * np.sin(theta)
    zpphi = lambda theta, phi :   0.

    # draw line
    xu = lambda phi : xpos(theta0, phi)
    yu = lambda phi : ypos(theta0, phi)
    zu = lambda phi : zpos(theta0, phi)
    line3d(phi_range, xu, yu, zu, newfig = False)

    xvelo, yvelo, zvelo = xpphi(theta0, phi0), ypphi(theta0, phi0), zpphi(theta0, phi0)
    arrow3d(x0, y0, z0, xvelo, yvelo, zvelo)

    # draw line
    xv = lambda theta : xpos(theta, phi0)
    yv = lambda theta : ypos(theta, phi0)
    zv = lambda theta : zpos(theta, phi0)
    line3d(theta_range, xv, yv, zv, newfig = False)

    xvelo, yvelo, zvelo = xptheta(theta0, phi0), yptheta(theta0, phi0), zptheta(theta0, phi0)
    arrow3d(x0, y0, z0, xvelo, yvelo, zvelo)

    return fig, ax
