
import numpy as np


def echarge(q, x0, y0):
    V  = lambda x, y: q       /((x-x0)**2 + (y-y0)**2)**(1/2)
    Ex = lambda x, y: q*(x-x0)/((x-x0)**2 + (y-y0)**2)**(3/2)
    Ey = lambda x, y: q*(y-y0)/((x-x0)**2 + (y-y0)**2)**(3/2)
    return V, Ex, Ey

def edipole(q, x0, y0, x1, y1):
    V1, Ex1, Ey1 = echarge( q, x0, y0)
    V2, Ex2, Ey2 = echarge(-q, x1, y1)
    V  = lambda x, y : V1(x, y)  + V2(x, y)
    Ex = lambda x, y : Ex1(x, y) + Ex2(x, y)
    Ey = lambda x, y : Ey1(x, y) + Ey2(x, y)
    return V, Ex, Ey

def esystem(qs):
    Vs = [echarge(*qi) for qi in qs]
    V  = lambda x, y : sum([Vi[0](x, y) for Vi in Vs])
    Ex = lambda x, y : sum([Vi[1](x, y) for Vi in Vs])
    Ey = lambda x, y : sum([Vi[2](x, y) for Vi in Vs])
    return V, Ex, Ey
