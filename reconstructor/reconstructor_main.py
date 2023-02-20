import vtk

from tool.utils import *


class AntiRecons(object):
    def __init__(self):
        # TODO: landmark info -> plane 가상평면 생성
        self.landmarkInfo = dict()

    def run(self, verts, polydata, DEBUG=False):
        """
         renconstruct 3d pointclouds to closed mesh
        """

        bmin, bmax = self._get_box_size(verts)
        new_verts, new_faces = self.build_mesh(bmin, bmax)

        if DEBUG:
            new = self.vert_face_to_polydata(new_verts, new_faces)
            new_actor = vtk_utils.polydata2actor(new)
            poly_actor = vtk_utils.polydata2actor(polydata)

            vtk_utils.show_actors([new_actor, poly_actor, vtk_utils.get_axes(10)])

        locator = vtk.vtkCellLocator()
        locator.SetDataSet(polydata)
        locator.BuildLocator()

        verts = self.projection_mesh(new_verts, locator)
        closed_mesh = self.closing_mesh(verts, new_faces, bmin, bmax)

        if DEBUG:
            vtk_utils.show_actors([vtk_utils.polydata2actor(closed_mesh)])

        return closed_mesh

    def _get_box_size(self, verts):
        bmin, bmax = verts.min(axis=0), verts.max(axis=0)

        return bmin, bmax

    def masked_points_to_mesh(self, images, depths, masks, DEBUG=False):
        polydatas = []
        for image, depth, mask in zip(images, depths, masks):

            masked_image, masked_depth = self._apply_mask(image, depth, mask)
            if DEBUG:
                pcd = make_point_cloud_from_rgbd(masked_image, masked_depth)
                o3d.visualization.draw_geometries([pcd])

            pcd_points = save_refined_mesh(masked_image, masked_depth, scale_factor=1e4)
            polydata = vtk_utils.read_stl("temp.stl")

            if DEBUG:
                vtk_utils.show_actors([vtk_utils.polydata2actor(polydata), vtk_utils.get_axes(10)])

            polydatas.append(polydata)

        return polydatas

    def _apply_mask(self, image, depth, mask):

        masked_image = image.copy()
        masked_depth = depth.copy()

        mask = mask > 0
        masked_image[mask == False] = 0
        masked_depth[masked_image[:, :, 0] == 0] = 0

        return masked_image, masked_depth

    def vert_face_to_polydata(self, v, f):
        verts = v
        faces = f

        points = vtk.vtkPoints()
        triangles = vtk.vtkCellArray()

        for i, tri in enumerate(faces):
            p1 = verts[tri[0]]
            p2 = verts[tri[1]]
            p3 = verts[tri[2]]
            points.InsertNextPoint(*p1)
            points.InsertNextPoint(*p2)
            points.InsertNextPoint(*p3)

            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, 3 * i + 0)
            triangle.GetPointIds().SetId(1, 3 * i + 1)
            triangle.GetPointIds().SetId(2, 3 * i + 2)
            triangles.InsertNextCell(triangle)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(triangles)

        return polydata

    def build_mesh(self, bmin, bmax, affine=None, delta=1e-3):
        """
        가상의 평면 방정식 생성
        """

        xmin, ymin, zmin = bmin
        xmax, ymax, zmax = bmax

        numofvingrid = 1.0 / delta

        numofgrid_x = (xmax - xmin) + 0.05
        numofgrid_y = (ymax - ymin) + 0.05

        outputVertices = []
        outputFaces = []
        outputInd = []

        for i in range(int(numofgrid_y * numofvingrid)):
            verticesinput_x = []
            for j in range(int(numofgrid_x * numofvingrid)):
                v = (xmin + j * delta + 1e-4, ymin + i * delta + 1e-4, zmax)
                # v = np.array([xmin + j * delta + 1e-4, ymin + i * delta + 1e-4, 0, 1])
                # v = np.dot(affine, v.T)[:3].tolist()

                outputVertices.append(v)

                # point = np.array([[i*scale, j*scale, 0, 1]])
                # plane_points.append(tuple(np.dot(affine, point.T)[:3].tolist()))
                verticesinput_x.append(len(outputVertices) - 1)
            outputInd.append(verticesinput_x)

        for i in range(int(numofgrid_y * numofvingrid) - 1):
            for j in range(len(outputInd[i]) - 1):
                outputFaces.append((outputInd[i][j], outputInd[i][j + 1], outputInd[i + 1][j]))
                outputFaces.append((outputInd[i + 1][j + 1], outputInd[i + 1][j], outputInd[i][j + 1]))

        return outputVertices, outputFaces

    def closing_mesh(self, verts, faces, bmin, bmax, affine=None, delta=1e-3):
        """
        최종 closed mesh 출력 ( blockout mesh )
        """
        numofvingrid = 1.0 / delta

        xmin, ymin, zmin = bmin
        xmax, ymax, zmax = bmax

        numofgrid_x = (xmax - xmin) + 0.05
        numofgrid_y = (ymax - ymin) + 0.05

        newInd = []

        for i in range(int(numofgrid_y * numofvingrid)):
            verticesinput_x = []
            for j in range(int(numofgrid_x * numofvingrid)):
                v = (xmin + j * delta + 1e-4, ymin + i * delta + 1e-4, zmax)
                # v = np.array([xmin + j * delta + 1e-4, ymin + i * delta + 1e-4, 0, 1])
                # v = np.dot(affine, v.T)[:3].tolist()
                verts.append(v)
                verticesinput_x.append(len(verts) - 1)
            newInd.append(verticesinput_x)

        for i in range(int(numofgrid_y * numofvingrid) - 1):
            for j in range(len(newInd[i]) - 1):
                faces.append((newInd[i][j], newInd[i][j + 1], newInd[i + 1][j]))
                faces.append((newInd[i + 1][j + 1], newInd[i + 1][j], newInd[i][j + 1]))

        closed_polydata = self.vert_face_to_polydata(verts, faces)

        return closed_polydata

    def smoothing_mesh(self, pd, iterations=15):

        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(pd)
        smoothFilter.SetNumberOfIterations(iterations)
        smoothFilter.SetRelaxationFactor(0.1)
        smoothFilter.FeatureEdgeSmoothingOff()
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.Update()

        normalGenerator = vtk.vtkPolyDataNormals()
        normalGenerator.SetInputConnection(smoothFilter.GetOutputPort())
        normalGenerator.ComputePointNormalsOn()
        normalGenerator.ComputeCellNormalsOn()
        normalGenerator.Update()

        return normalGenerator.GetOutput()

    @timefn2
    def projection_mesh(self, vertices, locator, delta=1e-8):

        i = 0
        for vs in vertices:
            isHit = False

            p1 = list(vs)
            p2 = list(vs)
            p2[2] = -150

            isexist, pose = self.find_intersection(p1, p2, locator)

            while (isexist):
                isHit = True
                vs = pose
                pre_z = pose[2] - 1
                p1 = [vs[0], vs[1], pre_z]
                isexist, pose = self.find_intersection(p1, p2, locator)

            if isHit == False:
                temp = list(vs)
                temp[2] -= delta
                vs = temp
            vertices[i] = vs

            i = i + 1

        return vertices

    def find_intersection(self, p1, p2, locator: vtk.vtkCellLocator):

        t = vtk.mutable(0)
        subId = vtk.mutable(0)
        tol = 0.001
        pose = [0, 0, 0]
        pcoords = [0, 0, 0]
        isexist = locator.IntersectWithLine(p1, p2, tol, t, pose, pcoords, subId)

        return isexist, pose

