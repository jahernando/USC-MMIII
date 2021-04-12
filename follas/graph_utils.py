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
from matplotlib.collections import PolyCollection
matplotlib.style.use('ggplot')

#figsize = 5, 3.8
#figsize = 7, 4.4 
figsize = 8, 6
cmap    = 'hot'

xrange = (-3., 3., 100)
trange = ( 0., 1., 100)

def fun1d(fun, xrange = xrange, newfig = True, label = ""):
    """ draw a 1d function
    parameters:
        fun   : (function)     real function
        xrange: (float, float, int) (a, b, nbins) partition of the interval (a, b) in nbins
    """
    xs   = np.linspace(*xrange)
    fig = plt.figure() if newfig else plt.gcf(); ax = plt.gca()
    ax.plot(xs, fun(xs), label = label)
    if (label != ''):
        ax.legend()
    ax.set_xlabel('$x$'); ax.set_ylabel('f(x)')
    return fig, ax


def meshgrid(xrange, yrange):
    xs  = np.linspace(*xrange)
    ys  = np.linspace(*yrange)
    xms, yms = np.meshgrid(xs, ys)
    return xms, yms

def zfun(fun, xms, yms, condition = None, zlim = None, zzero = 0.):
    zms = fun(xms, yms)
    if (condition is not None):
        mask = condition(xms, yms)
        zms[~mask] = zzero
    if (zlim is not None):
        zmin, zmax = zlim
        zms[ zms < zmin] = zmin
        zms[ zms > zmax] = zmax
    return zms

def graph(fun, xrange = xrange, yrange = xrange, condition = None,
          zlim = None, newfig = True, **kargs):
    """ draws a funcion
    parameters:
        fun    : (function)    x,y function to draw
        xrange : (a, b, nbins) x-partition of the interval (a, b) in nbins
        yrange : (a, b, nbins) y-partition of the interval (a, b) in nbins
        zlim   : (float, float) zlimits
        newfig : (bool)     False is use existing figure
    returns:
    """
    xms, yms = meshgrid(xrange, yrange)
    zms      = zfun(fun, xms, yms, condition, zlim)
    fig = plt.figure(figsize=figsize) if newfig else plt.gcf()
    ax = plt.gca(projection='3d')
    sf  = ax.plot_surface(xms, yms, zms, cmap=cmap, **kargs)
    if (zlim is not None): ax.set_zlim3d(*zlim)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); #ax.set_aspect('equal')
    if (newfig): fig.colorbar(sf)
    return fig, ax


def riemann_sum(fun, xrange, yrange, condition = None):
    xms, yms = meshgrid(xrange, yrange)
    zms      = zfun(fun, xms, yms, condition)
    norm = plt.Normalize()
    cmap = plt.cm.jet(norm(zms.ravel()))

    z0s = np.zeros_like(zms.ravel())
    xwidth, ywidth  = xms[0][1] - xms[0][0], yms[1][0] - yms[0][0]
    rsum = np.sum(zms) * xwidth * ywidth
    return rsum


def bars3d(fun, xrange = xrange, yrange = xrange, condition = None, zlim = None,
          newfig = True, acolor = 'z'):

    xms, yms = meshgrid(xrange, yrange)
    zms      = zfun(fun, xms, yms, condition, zlim)
    norm = plt.Normalize()
    cmap = plt.cm.jet(norm(zms.ravel()))

    z0s = np.zeros_like(zms.ravel())
    xwidth, ywidth  = xms[0][1] - xms[0][0], yms[1][0] - yms[0][0]

    fig = plt.figure(figsize = figsize) if newfig else plt.gcf()
    ax = plt.gca(projection='3d')
    crange  = zms
    if (acolor == 'x'): crange = xms
    if (acolor == 'y'): crange = yms
    norm = plt.Normalize()
    cmap = plt.cm.jet(norm(crange.ravel()))

    ax.bar3d(xms.ravel(), yms.ravel(), z0s.ravel(), xwidth, ywidth, zms.ravel(),
    shade=True, color = cmap)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); #ax.set_aspect('equal')

    return fig, ax


    return fig, ax

def contour(fun, xrange = xrange , yrange = xrange, contours = 20, condition = None,
            zlim = None, fill = True, newfig = True):
    xms, yms = meshgrid(xrange, yrange)
    zms      = zfun(fun, xms, yms, condition, zlim)
    fig = plt.figure(figsize=figsize) if newfig else plt.gcf()
    ax = plt.gca()
    if (fill):
        c0 = ax.contourf(xms, yms, zms, contours, alpha=0.8, cmap=cmap)
    sf  = ax.contour(xms, yms, zms, contours, cmap=cmap)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); #ax.set_aspect('equal')
    fig.colorbar(sf)
    return fig, ax
    return

def quiver2d(fx, fy, xrange = xrange, yrange = xrange, newfig = False):
    xms, yms = meshgrid(xrange, yrange)
    vmx      = fx(xms, yms)
    vmy      = fy(xms, yms)
    fig = plt.figure(figsize=figsize) if newfig else plt.gcf()
    ax = plt.gca()
    c0 = ax.quiver(xms, yms, vmx, vmy, alpha=0.8)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); #ax.set_aspect('equal')
    return fig, ax
    return


def quiver2d_in_line2d(fx, fy, cx, cy, trange = xrange, newfig = False,
                        color = 'red', alpha = 0.5):

    fig = plt.figure(figsize=figsize) if newfig else plt.gcf()
    ax = fig.gca()

    # Make the grid
    t = np.linspace(*trange)

    x , y  = cx(t), cy(t)
    vx, vy = fx(x, y), fy(x, y)

    ax.quiver(x, y, vx, vy, alpha = alpha, color = color)
    return


def quiver3d(fx, fy, fz, xrange = xrange, yrange = xrange, zrange = xrange,
              newfig = False, color = 'r'):
    fig = plt.figure(figsize=figsize) if newfig else plt.gcf()
    ax = fig.gca(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.linspace(*xrange),
                          np.linspace(*yrange),
                          np.linspace(*zrange))

                          # Make the direction data for the arrows
    u = fx(x, y, z)
    v = fy(x, y, z)
    w = fz(x, y, z)

    ax.quiver(x, y, z, u, v, w,  alpha = 0.8, length = 0.3, normalize = True, color = color)

    return


def quiver3d_in_wfsurface(fx, fy, fz, sx, sy, sz, urange = xrange, vrange = xrange,
                         newfig = False, alpha = 0.8, color = 'r'):
    fig = plt.figure(figsize=figsize) if newfig else plt.gcf()
    ax = fig.gca(projection='3d')

    # Make the grid
    u, v  = np.meshgrid(np.linspace(*urange), np.linspace(*vrange))

    x , y ,  z = sx(u, v), sy(u, v), sz(u, v)
    vx, vy, vz = fx(x, y, z), fy(x, y, z), fz(x, y, z)

    ax.quiver(x, y, z, vx, vy, vz,  alpha = alpha, length = 0.3, normalize = True, color = color)

    return


def quiver3d_in_line3d(fx, fy, fz, cx, cy, cz, trange = xrange,
                         newfig = False, alpha = 0.8, color = 'r'):
    fig = plt.figure(figsize=figsize) if newfig else plt.gcf()
    ax = fig.gca(projection='3d')

    # Make the grid
    t = np.linspace(*trange)

    x , y ,  z = cx(t), cy(t), cz(t)
    vx, vy, vz = fx(x, y, z), fy(x, y, z), fz(x, y, z)

    ax.quiver(x, y, z, vx, vy, vz,  alpha = alpha, length = 0.3, normalize = True, color = color)

    return

def dot(x0, y0, color = 'black'):
    ax = plt.gca()
    ax.plot(x0, y0, color= color, marker='*')
    return


def segment( p0 , p1, color = 'black'):
    x0, y0 = p0
    x1, y1 = p1
    ax = plt.gca()
    ax.plot((x0, x1), (y0, y1), color= color, alpha = 0.5)
    return

def square( p0, xside, yside, color = 'black'):
    x0, y0 = p0
    segment( (x0        , y0), (x0 + xside, y0), color = color)
    segment( (x0 + xside, y0), (x0 + xside, y0 + yside), color = color)
    segment( (x0        , y0), (x0        , y0 + yside), color = color)
    segment( (x0, y0 + yside), (x0 + xside, y0 + yside), color = color)
    return


def arrow(x0, y0, vx, vy, head = 0.3, color = 'black'):
    ax = plt.gca()
    vv = np.sqrt(vx*vx + vy*vy)
    if (vv == 0.):
        ax.plot(x0, y0, color = color, marker='*')
    else:
        vx, vy = (1-head)*vx, (1-head)*vy
        ax.arrow(x0, y0, vx, vy, head_width=head, head_length=head,
                color = color, lw=2)
    return ax


def arrow3d(x0, y0, z0, vx, vy, vz, color = 'black', head = 0.0):
    ax = plt.gca(projection='3d')
    vv = np.sqrt(vx*vx + vy*vy + vz*vz)
    if (vv == 0.):
        ax.plot(x0, y0, z0, color='black', marker='*')
    else:
        vx, vy, vz = (1-head)*vx, (1-head)*vy, (1-head)*vz
        ax.quiver(x0, y0, z0, vx, vy, vz, color = color, lw=2)
    return ax


def graph_section(fun, x0, y0, vx, vy, length = 5., sign = +1.):
    xs  = np.linspace(-length, length, 100)
    ys  = np.linspace(-length, length, 100)
    xms, yms = np.meshgrid(xs, ys)
    zms = fun(xms, yms)
    dvx, dvy = xms - x0, yms - y0
    dd  = dvy * vx - dvx * vy
    zmin = np.min(zms)
    zms[sign*dd >= 0.] = zmin
    fig = plt.figure(figsize=(8, 3.8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    sf = ax.plot_surface(xms, yms, zms, cmap = cmap)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); #ax.set_aspect('equal')
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


def line2d(funx, funy, trange = trange, color = 'blue', newfig = True):
    ts = np.linspace(*trange)
    xs, ys = funx(ts), funy(ts)
    fig = plt.figure(figsize = figsize) if newfig else plt.gcf()
    ax  = plt.gca()
    ax.plot(xs, ys, color = color)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); #ax.set_aspect('equal')
    return fig, ax


def line3d(funx, funy, funz, trange = xrange,
           newfig = True, color = 'blue', alpha = 0.8):
    ts = np.linspace(*trange)
    xs, ys, zs = funx(ts), funy(ts), funz(ts)
    fig = plt.figure(figsize = figsize) if newfig else plt.gcf()
    ax  = plt.gca(projection='3d')
    ax.plot(xs, ys, zs, color = color, lw = 2, alpha = 0.8)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); #ax.set_aspect('equal')
    return fig, ax


def wfsurface(funx, funy, funz, urange = xrange, vrange = xrange,
              newfig = True, color = 'r', alpha = 1.):
    us = np.linspace(*urange)
    vs = np.linspace(*vrange)
    ums, vms = np.meshgrid(us, vs)
    xms, yms, zms = funx(ums, vms), funy(ums, vms), funz(ums, vms)
    fig = plt.figure(figsize = figsize) if newfig else plt.gcf()
    ax  = plt.gca(projection='3d')
    ax.plot_wireframe(xms, yms, zms, alpha = alpha, color = color)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); #ax.set_aspect('equal')
    return fig, ax



def wfmasterlines(funx, funy, funz, urange, vrange, ui = 0, vj = 0,
           color='b', alpha = 0.8):
    us = np.linspace(*urange)
    vs = np.linspace(*vrange)
    ums, vms = np.meshgrid(us, vs)
    xms, yms, zms = funx(ums, vms), funy(ums, vms), funz(ums, vms)
    u0, v0 = np.ones(len(us)) * us[ui], np.ones(len(vs)) * vs[vj]
    ax = plt.gca(projection='3d')
    ulx = lambda u : funx(u, v0)
    uly = lambda u : funy(u, v0)
    ulz = lambda u : funz(u, v0)
    line3d(ulx, uly, ulz, urange, newfig=False, color = color, alpha = alpha)
    vlx = lambda v : funx(u0, v)
    vly = lambda v : funy(u0, v)
    vlz = lambda v : funz(u0, v)
    line3d(vlx, vly, vlz, vrange, newfig=False, color = color, alpha = alpha)
    return


def wfaxis(funx, funy, funz, urange, vrange, ui = 0, vj = 0,
           color='black', alpha = 0.8):
    i, j = ui, vj
    us = np.linspace(*urange)
    vs = np.linspace(*vrange)
    ums, vms = np.meshgrid(us, vs)
    xms, yms, zms = funx(ums, vms), funy(ums, vms), funz(ums, vms)
    x0, y0, z0 = xms[j][i], yms[j][i], zms[j][i]
    dxj, dyj, dzj = xms[j+1][i] - xms[j][i], yms[j+1][i] - yms[j][i], zms[j+1][i] - zms[j][i]
    dxi, dyi, dzi = xms[j][i+1] - xms[j][i], yms[j][i+1] - yms[j][i], zms[j][i+1] - zms[j][i]
    nx =  dyi * dzj - dyj * dzi
    ny = -dxi * dzj + dxj * dzi
    nz =  dxi * dyj - dxj * dyi
    mod = lambda x, y, z: np.sqrt(x*x + y*y + z*z)
    nn = np.sqrt(mod(nx, ny, nz))
    ax = plt.gca(projection='3d')
    ax.quiver(x0, y0, z0, dxi, dyi, dzi, color = color, lw=2, alpha = alpha)
    ax.quiver(x0, y0, z0, dxj, dyj, dzj, color = color, lw=2, alpha = alpha)
    ax.quiver(x0, y0, z0,  nx/nn,  ny/nn,  nz/nn, color = color, lw=2, alpha = alpha)
    return xms, yms, zms



def wfsurface2d(xfun, yfun, urange, vrange, newfig = False,
                xlabel = '$x$', ylabel = '$y$', color = 'r'):
    us = np.linspace(*urange)
    vs = np.linspace(*vrange)
    ums, vms = np.meshgrid(us, vs)
    nus, nvs = ums.shape
    xms = xfun(ums, vms)
    yms = yfun(ums, vms)
    fig = plt.figure(figsize = figsize) if newfig else plt.gcf()
    ax  = plt.gca()
    for i in range(nus):
        ax.plot(xms[i][:], yms[i][:], color = color, alpha = 0.8)
    xms = xms.transpose()
    yms = yms.transpose()
    for i in range(nvs):
        ax.plot(xms[i][:], yms[i][:], color = color, alpha = 0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); #ax.set_aspect('equal')
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


def int_fscalar_line(fc, cx, cy, trange = trange, newfig = False):
    ts = np.linspace(*trange)
    xs, ys = cx(ts), cy(ts)
    zs     = fc(xs, ys)
    fig = plt.figure(figsize = figsize) if newfig else plt.gcf()
    ax  = plt.gca(projection='3d')
    ax.plot(xs, ys, zs, color = 'r')
    ax.plot(xs, ys, 0.*zs, color = 'black', alpha = 0.5)
    for i in range(len(xs)):
        ax.plot( (xs[i], xs[i]), (ys[i], ys[i]), (0., zs[i]), color = 'r', alpha = 0.5)
#ax.view_init(azim=60.)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); #ax.set_aspect('equal');
    return

def int_fvect_line(fx, fy, cx, cy, trange = trange, newfig = False,
                   surf = True):
    ts = np.linspace(*trange)
    xs, ys = cx(ts), cy(ts)
    dxs = np.array([xs[i+1] - xs[i] for i in range(len(xs)-1)])
    dys = np.array([ys[i+1] - ys[i] for i in range(len(ys)-1)])
    fxs = fx(xs, ys)
    fys = fy(xs, ys)
    vals = zip(fxs[:-1], fys[:-1], dxs, dys)
    ds  = np.array([np.sqrt(dxi * dxi + dyi * dyi)
                    for dxi, dyi in zip(dxs, dys)])
    zzs = np.array([fxi * dxi + fyi * dyi for fxi, fyi, dxi, dyi in vals])
    intval = np.sum(zzs)
    zs  = np.array([zi/dsi for zi, dsi in zip(zzs, ds)])
    fig = plt.figure(figsize = figsize) if newfig else plt.gcf()
    #xs, ys = xs[:-1], ys[:-1]
    if (surf):
        ax  = plt.gca()
        ax.plot(xs, ys, color = 'black', alpha = 0.5)
        #vers = list(zip(xs, fxs))
        #poly = PolyCollection(vers)
        #ax.add_collection(poly)
        #fxm = 0.5*(fxs[:-1] + fxs[1:])
        for i in range(len(xs)-1):
            color = 'b' if dxs[i] > 0. else 'r'
            ifx = 0.5*(fxs[i+1] + fxs[i])
            ax.plot( (xs[i]  , xs[i])  , (0. , ifx ), color = color, alpha = 0.4)
            ax.plot( (xs[i+1], xs[i+1]), (0. , ifx ), color = color, alpha = 0.4)
            ax.plot( (xs[i]  , xs[i+1]), (ifx, ifx ), color = color, alpha = 0.4)
            color = 'b' if dys[i] > 0. else 'r'
            ify = 0.5*(fys[i+1] + fys[i])
            ax.plot( (0. , ify), (ys[i]  , ys[i])  , color = color, alpha = 0.4)
            ax.plot( (0. , ify), (ys[i+1], ys[i+1]), color = color, alpha = 0.4)
            ax.plot( (ify, ify), (ys[i]  , ys[i+1]), color = color, alpha = 0.4)
        #ax.
        #ax.ax.plot( (0., ify), (ys[i], ys[i]), color = color, alpha = 0.4)
    #ax.
        #ax.view_init(azim=60.)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); # ax.set_aspect('equal');
    else:
        ax  = plt.gca(projection='3d')
        xs, ys = xs[:-1], ys[:-1]
        ax.plot(xs, ys, zs, color = 'r')
        ax.plot(xs, ys, 0.*zs, color = 'black', alpha = 0.5)
        for i in range(len(xs)):
            ax.plot( (xs[i], xs[i]), (ys[i], ys[i]), (0., zs[i]), color = 'r', alpha = 0.5)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); #ax.set_aspect('equal');
    return intval


def sphere_plane(theta0,  phi0, length = 0.5):
    if (theta0 == np.pi/2.):
        print('Can not draw tangent plane at theta pi/2 ', theta0)
        return

    # draw sphere
    theta_range = 0., np.pi, 20
    phi_range   = 0., 2.*np.pi, 20
    r = 1.
    xpos = lambda theta, phi : r * np.cos(phi) * np.sin(theta)
    ypos = lambda theta, phi : r * np.sin(phi) * np.sin(theta)
    zpos = lambda theta, phi : r               * np.cos(theta)
    fig, ax = wfsurface(xpos, ypos, zpos, theta_range, phi_range, color='orange');

    # draw plane
    fpx  = lambda x, y, z: 2*x
    fpy  = lambda x, y, z: 2*y
    fpz  = lambda x, y, z: 2*z
    x0, y0, z0 = xpos(theta0, phi0), ypos(theta0, phi0), zpos(theta0, phi0)
    fpx0, fpy0, fpz0 = fpx(x0, y0, z0), fpy(x0, y0, z0), fpz(x0, y0, z0)
    xplane = lambda x, y : x
    yplane = lambda x, y : y
    zplane = lambda x, y : z0 - (fpx0 * (x-x0) + fpy0 * (y - y0))/fpz0
    x_range = x0 - length, x0 + length, 10
    y_range = y0 - length, y0 + length, 10
    wfsurface(xplane, yplane, zplane, x_range, y_range, newfig = False)

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
    line3d(xu, yu, zu, phi_range, newfig = False)

    xvphi, yvphi, zvphi = xpphi(theta0, phi0), ypphi(theta0, phi0), zpphi(theta0, phi0)
    arrow3d(x0, y0, z0, xvphi, yvphi, zvphi)

    # draw line
    xv = lambda theta : xpos(theta, phi0)
    yv = lambda theta : ypos(theta, phi0)
    zv = lambda theta : zpos(theta, phi0)
    line3d(xv, yv, zv, theta_range, newfig = False)

    xvtheta, yvtheta, zvtheta = xptheta(theta0, phi0), yptheta(theta0, phi0), zptheta(theta0, phi0)
    arrow3d(x0, y0, z0, xvtheta, yvtheta, zvtheta)

    grad   = (fpx0 ,      fpy0, fpz0)
    vphi   = (xvphi,     yvphi, zvphi)
    vtheta = (xvtheta, yvtheta, zvtheta)

    return fig, ax, grad, vphi, vtheta
