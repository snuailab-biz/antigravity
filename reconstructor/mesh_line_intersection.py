import os

import numpy as np
import vtk

from tool import vtk_utils

# FIXME : set test-file
path = '../temp.stl'
assert os.path.exists(path), FileNotFoundError(path)
pd = vtk_utils.read_stl(path)
# [x1, y1, z1], [x2, y2, z2]
bounds = np.array(pd.GetBounds()).reshape([-1, 2]).T
bmin, bmax = bounds
ctr = (bmax + bmin) / 2
dy = np.linspace(bmin[1], bmax[1], 10)
dz = bmax[2] - bmin[2]
scale = 0.1
z1 = bmin[2] - scale * dz
z2 = bmax[2] + scale * dz
cx = ctr[0]

locator = vtk.vtkCellLocator()
locator.SetDataSet(pd)
locator.BuildLocator()


def find_intersection(p1, p2):
    """
    https://vtk.org/Wiki/VTK/Examples/Python/DataManipulation/LineOnMesh
    """
    # assert isinstance(p1, )
    t = vtk.mutable(0)
    subId = vtk.mutable(0)
    tol = 0.001
    pose = [0, 0, 0]
    pcoords = [0, 0, 0]
    isexist = locator.IntersectWithLine(p1, p2, tol, t, pose, pcoords, subId)
    return isexist, pose


actor = vtk_utils.polydata2actor(pd)
show_actors = []
show_actors.append(actor)
for iy in dy:
    p1 = [cx, iy, z2]
    p2 = [cx, iy, z1]

    isexist, pose = find_intersection(p1, p2)

    if isexist:
        p12 = np.stack([p1, p2], axis=0)
        p12_line = vtk_utils.create_curve_actor(p12)
        sp = vtk_utils.create_sphere([pose], .2)

        show_actors.append(p12_line)
        show_actors.extend(sp)

vtk_utils.show_actors([*show_actors, vtk_utils.get_axes(10)])
