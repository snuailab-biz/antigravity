import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import vtk
from scipy.ndimage import gaussian_filter
from vtkmodules.util import numpy_support
from vtkmodules.util.colors import *


def show_plots(points):
    """
    :param points:[N, 3]
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pts = points.T
    ax.plot(*pts, '*')
    plt.show()


def compute_boundary_edge(polydata):
    featureEdges = vtk.vtkFeatureEdges()
    featureEdges.SetInputData(polydata)
    featureEdges.BoundaryEdgesOn()
    featureEdges.FeatureEdgesOff()
    featureEdges.ManifoldEdgesOff()
    featureEdges.NonManifoldEdgesOff()
    featureEdges.Update()
    return featureEdges.GetOutput()


def compute_curvature(polydata_or_actor):
    if isinstance(polydata_or_actor, vtk.vtkActor):
        polydata = polydata_or_actor.GetMapper().GetInput()
    elif isinstance(polydata_or_actor, vtk.vtkPolyData):
        polydata = polydata_or_actor
    curvaturesFilter = vtk.vtkCurvatures()
    curvaturesFilter.SetInputData(polydata)
    curvaturesFilter.SetCurvatureTypeToMinimum()
    curvaturesFilter.SetCurvatureTypeToMaximum()
    curvaturesFilter.SetCurvatureTypeToGaussian()
    curvaturesFilter.SetCurvatureTypeToMean()
    curvaturesFilter.Update()

    return curvaturesFilter.GetOutput()


def get_axes(scales=None):
    v = 50 if scales is None else scales
    import vtk
    axes = vtk.vtkAxesActor()
    t = vtk.vtkTransform()
    t.Scale(v, v, v)
    axes.SetUserTransform(t)

    return axes


def get_transform_axes(afm, scales=50):
    axes = get_axes(scales)
    post = afm
    pre = axes.GetUserTransform()
    concat_t = myTransform()
    concat_t.Concatenate(post)
    concat_t.Concatenate(pre)
    axes.SetUserTransform(concat_t)
    return axes


def create_points_actor(x, invert=False):

    # pt = x
    # N = x.shape[0]
    if invert:
        x = x[:, ::-1]

    pt = x
    # pt = np.concatenate([x, y])
    points = vtk.vtkPoints()
    vtkarray = numpy_support.numpy_to_vtk(pt, array_type=vtk.VTK_FLOAT)
    points.SetData(vtkarray)
    # points.SetData(vtkpoints)
    # for i in range(pt.shape[0]):
    #     points.InsertNextPoint(*tuple(pt[i]))
    verts = vtk.vtkCellArray()
    for i in range(x.shape[0]):
        verts.InsertNextCell(1)
        verts.InsertNextCell(i)

    point_polydata = vtk.vtkPolyData()
    point_polydata.SetPoints(points)
    point_polydata.SetVerts(verts)
    point_actor = polydata2actor(point_polydata)

    point_actor.GetProperty().SetColor(tuple(np.random.uniform(0, 1, 3)))
    return point_actor


def create_curve_actor(x, closed=False):
    line_polydata = vtk.vtkPolyData()

    points = vtk.vtkPoints()

    vtkpoints = numpy_support.numpy_to_vtk(x, array_type=vtk.VTK_FLOAT)
    points.SetData(vtkpoints)
    N = x.shape[0]
    lines = vtk.vtkCellArray()
    for i in range(x.shape[0]-1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i+1)
        lines.InsertNextCell(line)

    if closed:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, x.shape[0]-1)
        line.GetPointIds().SetId(1, 0)
        lines.InsertNextCell(line)

    line_polydata.SetPoints(points)
    line_polydata.SetLines(lines)

    line_actor = polydata2actor(line_polydata)
    line_actor.GetProperty().SetColor(tuple(np.random.uniform(0, 1, 3)))
    return line_actor


def keypress_event(obj, event):
    key = obj.GetKeySym()
    print(key)
    if key == "d":
        save_dir = "test_image"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        imagefilter = vtk.vtkWindowToImageFilter()

        imagefilter.SetInput(obj.GetRenderWindow())
        # imagefilter.SetM
        imagefilter.ReadFrontBufferOff()
        imagefilter.SetInputBufferTypeToRGBA()
        imagefilter.Update()

        lt = time.localtime()

        str_time = "{}{}{}".format(lt.tm_hour, lt.tm_min, lt.tm_sec)

        writer = vtk.vtkPNGWriter()
        writer.SetFileName("test_image/{}.png".format(str_time))
        writer.SetInputConnection(imagefilter.GetOutputPort())
        writer.Write()

        obj.GetRenderWindow().GetRenderers().GetFirstRenderer().ResetCamera()
        obj.Render()


    elif key in ["Left", "Up", "Right", "Down", "1", "2", "3", "4"]:

        ren = obj.GetRenderWindow().GetRenderers().GetFirstRenderer()
        props = [a for a in ren.GetViewProps()]
        prop = props[0]

        ctr = prop.GetCenter()
        bounds = prop.GetBounds()

        pos = [ctr[0] , ctr[1] , ctr[2]]
        if key == "Left":
            pos[0] = pos[0] - bounds[1] * 3
        elif key == "Right":
            pos[0] = pos[0] + bounds[1] * 3
        elif key == "Up":
            pos[1] = pos[1] + bounds[1]*3
        elif key == "Down":
            pos[1] = pos[1] - bounds[1]*3
        elif key.isdigit():
            key_int = int(key)
            dl = 100
            cam_list = (
                (0, dl, 0),
                (0, -dl, 0),
                (dl, 0, 0),
                (-dl, 0, 0),
                (0, 0, dl),
                (0, 0, -dl),
            )
            viewup = 1
            if 0 <= key_int < 6:
                pos = cam_list[key_int]
                viewup = 1
        else:
            pass

        cam = ren.GetActiveCamera()

        cam.SetViewUp(0, 0, viewup)
        cam.SetPosition(*pos)

        obj.Render()


def _get_opacity_property(threshold=140, opacity_scalar=1.0):
    import vtk
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-3024, 0)
    # opacity.AddPoint(0, 0)
    opacity.AddPoint(0, 0.00)
    opacity.AddPoint(threshold*0.8, 0.0)
    opacity.AddPoint(threshold, opacity_scalar)

    return opacity

def convert_numpy_2_vtkmarching(volume_array, threshold, radius=1., dev=2.):
    vtkimage = convert_numpy_vtkimag(volume_array)
    pd = convert_voxel_to_polydata(vtkimage, threshold, radius, dev)
    return polydata2actor(pd)


def convert_vtkimag_numpy(vtk_image_array:vtk.vtkImageData):
    numpy_array = numpy_support.vtk_to_numpy(vtk_image_array.GetPointData().GetScalars())
    return numpy_array.reshape(vtk_image_array.GetDimensions()[::-1])


def convert_numpy_vtkimag(volume_array):
    vtk_array = numpy_support.numpy_to_vtk(volume_array.astype(np.uint16).ravel(), array_type=vtk.VTK_UNSIGNED_SHORT)

    pd = vtk.vtkImageData()
    pd.GetPointData().SetScalars(vtk_array)
    pd.SetDimensions(volume_array.shape[::-1])
    return pd


def show_volume(imgdata):
    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputData(imgdata)
    volumeMapper.SetBlendModeToComposite()

    volume = vtk.vtkVolume()

    volume.SetMapper(volumeMapper)
    volume.SetProperty(_get_normal_property())
    volume.Update()

    ren = vtk.vtkRenderer()
    ren.AddVolume(volume)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # ren.GetActiveCamera()->SetViewUp(0, 1, 0);
    ren.GetActiveCamera().SetFocalPoint(*volume.GetCenter())
    # ren.GetActiveCamera()->SetPosition(c[0], c[1], c[2]);

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()


def change_actor_color(actors, color):
    [a.GetProperty().SetColor(*color) for a in actors]


def get_aabb_cubes(bbox):
    """
    :param bbox: [N, 6]
    :return:
    """
    bbox = convert_box_norm2vtk(bbox)
    return [get_cube(b) for b in bbox]

def get_cube(bounds, color=None):
    """
    :param bounds: vtk format, (x1,x2,y1,y2,z1,z2)
    :return:
    """
    import vtk
    cube = vtk.vtkCubeSource()
    cube.SetBounds(*list(bounds))
    cube.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(cube.GetOutput())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(3)
    actor.GetProperty().LightingOff()
    color = color or tuple(np.random.uniform(0, 1, 3))
    actor.GetProperty().SetColor(color)

    return actor
#
def auto_refinement_mask(mask_volume):
    """
    after cropping bounds non-zero value,
    converting voxel to mesh unsing marching cube
    :param mask_volume:
    :return:
    """

    # box

    uniquevalue = np.unique(mask_volume)
    fgvalue = uniquevalue[uniquevalue > 0]
    # remove noise out of box
    # self.apply_box_mask(number)
    offset = 5
    max_inds = np.array(mask_volume.shape)-1
    crop_actors = []
    for number in fgvalue:
        # center of index
        bool_mask = mask_volume == number
        ix = np.stack(np.where(bool_mask), axis=-1)
        center = np.round(np.mean(ix, axis=0)).astype(np.int)

        # region growing in center of mass, remove other index
        umask = bool_mask.astype(np.uint8)
        region_mask = np.zeros_like(umask)

        inds = np.stack(np.where(bool_mask), axis=-1)
        p1 = inds.min(axis=0) - offset
        p2 = inds.max(axis=0) + offset
        p1 = np.clip(p1, 0, max_inds)
        p2 = np.clip(p2, 0, max_inds)
        z1, y1, x1 = p1
        z2, y2, x2 = p2
        cropmask = bool_mask[z1:z2, y1:y2, x1:x2]
        box = np.concatenate([p1, p2])

        # rendering update
        actor = convert_vtkvolume(cropmask, box, number, radius=1., dev=1.)
        crop_actors.append(actor)
        # show_actors(crop_actors)
    return crop_actors


def compute_cell_normal(polydata, splitting=False):
    normal_gen = vtk.vtkPolyDataNormals()
    normal_gen.SetInputData(polydata)
    normal_gen.ComputeCellNormalsOn()
    normal_gen.ComputePointNormalsOff()

    if splitting is False:
        normal_gen.SplittingOff()

    normal_gen.Update()
    normal_polydata = normal_gen.GetOutput()
    polydata.GetCellData().SetNormals(normal_polydata.GetCellData().GetNormals())


def compute_normal(polydata:vtk.vtkPolyData, splitting = False, norm_copy=True, recompute=False):
    # = normdata.GetPointData().GetNormals()
    norm = polydata.GetPointData().GetNormals()
    if norm is None or recompute:
        normal_gen = vtk.vtkPolyDataNormals()
        normal_gen.SetInputData(polydata)
        normal_gen.ComputePointNormalsOn()
        if splitting is False:
            normal_gen.SplittingOff()

        normal_gen.ComputeCellNormalsOff()
        normal_gen.Update()
        normal_polydata = normal_gen.GetOutput()
        polydata.GetPointData().SetNormals(normal_polydata.GetPointData().GetNormals())
        return normal_polydata
    else:
        if norm.GetNumberOfTuples() == 0:
            normal_gen = vtk.vtkPolyDataNormals()
            normal_gen.SetInputData(polydata)
            normal_gen.ComputePointNormalsOn()
            if splitting is False:
                normal_gen.SplittingOff()

            normal_gen.ComputeCellNormalsOff()
            normal_gen.Update()
            normal_polydata = normal_gen.GetOutput()
            polydata.GetPointData().SetNormals(normal_polydata.GetPointData().GetNormals())
        else:
            pass


        return polydata


def create_sphere(pts, size=None):
    """
    pts : [N, 3] array
    return : vtk spheres actor
    """
    size = size or 1.
    spheres = []
    for pt in pts:
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetRadius(size)

        sphereMapper = vtk.vtkDataSetMapper()
        sphereMapper.SetInputConnection(sphereSource.GetOutputPort())

        forwardSphere =  vtk.vtkActor()#vtk.vtkActor()
        forwardSphere.PickableOff()
        forwardSphere.SetMapper(sphereMapper)
        forwardSphere.SetPosition(*pt)
        forwardSphere.GetProperty().SetColor(*np.random.uniform(0, 1, 3))
        spheres.append(forwardSphere)

    return spheres


def apply_transform_actor(actor: vtk.vtkActor, t):
    afm = actor.GetUserTransform()
    if afm:
        pd = apply_transform_polydata(actor.GetMapper().GetInput(), afm)
    else:
        pd = actor.GetMapper().GetInput()
    tpd = apply_transform_polydata(pd, t)
    act = polydata2actor(tpd)
    return act


def apply_transform_polydata(polydata, transform):
    if isinstance(transform, np.ndarray):
        transform = myTransform(transform)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(polydata)
    transformFilter.Update()
    return transformFilter.GetOutput()


def volume_marching_labeling(numpy_array, labels, color_table, offset=5, sigma=0.2, smoothing=False):
    actors = []
    expand = np.array([offset]*6)
    expand = expand * np.array([-1, -1, -1, 1, 1, 1])
    max_inds = np.array(numpy_array.shape) -1
    max_inds = np.pad(max_inds, [0, 3], mode='wrap')
    min_inds = np.zeros_like(max_inds)

    for lab in labels:
        zyx = np.where(numpy_array == lab)
        zyx = np.stack(zyx, axis=1)
        if zyx.size > 0:
            p1, p2 = zyx.min(axis=0), zyx.max(axis=0)+1
            pbox = np.concatenate([p1, p2])
            pbox = pbox + expand
            pbox = np.clip(pbox, min_inds, max_inds)
            z1, y1, x1, z2, y2, x2 = pbox

            crop = (numpy_array[z1:z2, y1:y2, x1:x2] == lab).astype(np.int)
            if smoothing:
                crop = gaussian_filter(crop.astype(np.float), sigma)
            actor = convert_numpy_2_vtkmarching(crop*255, 123)
            t = vtk.vtkTransform()
            t.Translate(x1, y1, z1)
            actor.SetUserTransform(t)
            if lab > 10:
                c = color_table[lab%10]
            else:
                c = color_table[9]
            actor.GetProperty().SetColor(*c)
            actor.GetMapper().ScalarVisibilityOff()
            actors.append(actor)

    return actors


def volume_labeling(numpy_array, labels, color_table):
    """
    :param numpy_array: [D, H, W] , uint type
    :param labels: [N], labels, uint type
    :return:
    """
    # numpy_array = gaussian_filter(numpy_array, 0.9)
    factor = 100
    labels = np.asarray(labels)
    color_table = np.asarray(color_table)
    # assert labels.shape[0] == colors.shape[0]
    # numpy_array2 = gaussian_filter(numpy_array, 2.0, mode='mirror')
    vtk_array = numpy_support.numpy_to_vtk(numpy_array.astype(np.uint16).ravel() * factor, array_type=vtk.VTK_UNSIGNED_SHORT)
    # visulaize(numpy_array2*255, 100)

    pd = vtk.vtkImageData()
    pd.GetPointData().SetScalars(vtk_array)
    pd.SetDimensions(numpy_array.shape[::-1])

    volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper.SetInputData(pd)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)

    # property
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOn()

    thresh = np.unique(labels[labels > 0]).min()
    maxv = np.unique(labels[labels > 0]).max()
    color = vtk.vtkColorTransferFunction()

    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0, 0.00)
    for v in np.unique(labels):
        value = v if v in color_table else v % 10

        c = color_table[value]
        color.AddRGBPoint(v*factor, *tuple(c))

        opacity.AddPoint(thresh * factor, 0.85)
        volumeProperty.SetScalarOpacity(opacity)


    volumeProperty.SetColor(color)

    volume.SetProperty(volumeProperty)
    volume.Update()

    return volume


def compare_image(vol_image, mask_image, thres=.5):
    """
    :param vol_image: [D,H,W] 0~255
    :param mask_image:
    :return:
    """
    from common.common import apply_mask
    fig = plt.figure()
    for src, ma in zip(vol_image, mask_image):
        drawing = np.repeat(np.expand_dims(src, axis=-1), 3, 2).astype(np.uint8)
        apply_mask(drawing, ma, (0, 1, 0), thres)
        plt.cla()
        plt.imshow(drawing)
        plt.pause(0.02)
    plt.close('all')


def show_volume_list(volumes_list):
    vols = [numpyvolume2vtkvolume(np.squeeze(v)*255, 123, division=i+1) for i, v in
        enumerate(volumes_list)]
    return vols


def create_vector(norm, pts, scale=10, invert=False):

    line_polydata = vtk.vtkPolyData()
    if invert:
        pts = pts[:, ::-1]
        norm = norm[:, ::-1]
    start = pts
    end = pts + norm * scale

    points = vtk.vtkPoints()

    stack_points = np.concatenate([start, end], axis=0)

    vtkpoints = numpy_support.numpy_to_vtk(stack_points, array_type=vtk.VTK_FLOAT)
    points.SetData(vtkpoints)
    N = pts.shape[0]
    lines = vtk.vtkCellArray()
    for i in range(0, pts.shape[0]-1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, N+i)
        lines.InsertNextCell(line)

    line_polydata.SetPoints(points)
    line_polydata.SetLines(lines)
    return polydata2actor(line_polydata)
    # return line_polydata

def read_stl(path):
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', path))
    encode_path = path.encode('euc-kr') if hanCount > 0 else path

    reader = vtk.vtkSTLReader()
    reader.SetFileName(encode_path)
    reader.Update()

    return reader.GetOutput()


def write_stl(filename, polydata_or_actor):
    polydata = polydata_or_actor
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(filename)
    writer.SetFileTypeToBinary()
    writer.SetInputData(polydata)
    writer.Write()

def apply_transform_tensor(tensor, transform):
    """
    :param tensor:[N, 6]  3pose - 3orientation
    :param transform:
    :return:
    """
    pose, orient = tensor[:, :3], tensor[:, 3:]
    tpose = apply_trasnform_np(pose, transform)
    torient = apply_rotation(orient, transform)
    return np.concatenate([tpose, torient], axis=-1)

def apply_trasnform_np(pts, transform):
    return np.dot(pts, transform[:3, :3].T) + transform[:3, 3]

def apply_rotation(normals, transform):
    """
    :param normals: direction vector [N, 3]
    :param transform:
    :return:
    """
    return np.dot(normals, transform[:3, :3].T)

def numpyvolume2vtkvolume(numpy_array, threshold, division=1, opacity=0.8, color=None):
    vtk_array = numpy_support.numpy_to_vtk(numpy_array.astype(np.uint16).ravel(), array_type=vtk.VTK_UNSIGNED_SHORT)

    pd = vtk.vtkImageData()
    pd.GetPointData().SetScalars(vtk_array)
    pd.SetDimensions(numpy_array.shape[::-1])

    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputData(pd)
    volumeMapper.SetBlendModeToComposite()

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(_get_normal_property(threshold, division=division, opacity=opacity, colors=color))
    volume.Update()
    return volume

def change_opacity_vtkvolume(vtkvolume:vtk.vtkVolume, threshold, opacity=1.0, division=1, color=None):
    opacity_prop = _get_normal_property(threshold, division=division, opacity=opacity, colors=color)
    vtkvolume.SetProperty(opacity_prop)


def visulaize(volume_array, threshold, viewup=-1):
    volume = numpyvolume2vtkvolume(volume_array, threshold)

    ren = vtk.vtkRenderer()
    ren.AddVolume(volume)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    ren.GetActiveCamera().SetFocalPoint(*volume.GetCenter())

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.AddObserver("KeyPressEvent", keypress_event)
    iren.Initialize()
    iren.Start()


def _normalize(p):
    p[:] = p / np.linalg.norm(p)


def create_arrow(p1, p2):
    norm_x = p2 - p1
    length = np.linalg.norm(norm_x)
    _normalize(norm_x)

    norm_z = np.cross(norm_x, np.random.randn(3))

    _normalize(norm_z)

    norm_y = np.cross(norm_z, norm_x)

    mat = np.eye(4)
    mat[:3, :3] = np.stack([norm_x, norm_y, norm_z], axis=1)

    tmat = myTransform()
    tmat.set_from_numpy_mat(mat)
    transform = myTransform()
    transform.Translate(*tuple(p1))
    transform.Concatenate(tmat)
    transform.Scale(length, length, length)

    arrow_source = vtk.vtkArrowSource()
    arrow_source.Update()
    arrow = arrow_source.GetOutput()
    arrow_actor = polydata2actor(arrow)
    arrow_actor.SetUserTransform(transform)
    return arrow_actor


def create_arrow_polydata(p1, p2):
    norm_x = p2 - p1
    length = np.linalg.norm(norm_x)
    _normalize(norm_x)

    norm_z = np.cross(norm_x, np.random.randn(3))

    _normalize(norm_z)

    norm_y = np.cross(norm_z, norm_x)

    mat = np.eye(4)
    mat[:3, :3] = np.stack([norm_x, norm_y, norm_z], axis=1)

    tmat = myTransform()
    tmat.set_from_numpy_mat(mat)
    transform = myTransform()
    transform.Translate(*tuple(p1))
    transform.Concatenate(tmat)
    transform.Scale(length, length, length)

    arrow_source = vtk.vtkArrowSource()
    arrow_source.Update()
    arrow = arrow_source.GetOutput()

    return apply_transform_polydata(arrow, transform)


def convert_box_norm2vtk(boxes):
    """
    normal format (z1, y1, x1, z2, y2, x2)
    to vtk-format (x1, x2, y1, y2, z1, z2)

    :param boxes:
    :return:
    """
    boxes_vtk = np.empty_like(boxes)
    # # (x1, x2, y1, y2, z1, z2)

    boxes_vtk[:, ::2] = boxes[:, :3][:, ::-1]
    boxes_vtk[:, 1::2] = boxes[:, 3:][:, ::-1]
    return boxes_vtk


def polydata2voxelization_withpad(polydata, spacing, input_bounds=None, return_center=True, return_origin=False, padding=0.0, expandding=1.05, return_bounds=False):
    """
    :param polydata: vtkPolydata
    :param spacing: tuple or list of 3 (float)
    :return: voxel data and voxel_center
    """
    spacing = np.asarray(spacing)

    whiteImg = vtk.vtkImageData()
    actual_bounds = np.array(polydata.GetBounds()).reshape([-1, 2]).T

    # expand bounds, to protect voxel boundary truncation.
    if input_bounds is None:
        bmin, bmax = actual_bounds[0], actual_bounds[1]
        ctr = (bmax + bmin) / 2
        ext = (bmax - bmin) / 2
        fext = ext + padding
        emin, emax = ctr - fext, ctr + fext
        # (3, 2)
        bounds = np.stack([emin, emax], axis=-1)
    else:
        bounds = input_bounds

    dim = np.ceil((bounds[:, 1] - bounds[:, 0]) / np.array(spacing)).astype(np.uint)
    ones = np.ones([3], dtype=np.uint)
    dim = dim + ones
    # dim = np.ceil((bounds[:, 1] - bounds[:, 0]) / np.array(spacing)).astype(np.int)
    min_bound = bounds[:, 0]
    origin = bounds[:, 0]
    whiteImg.SetDimensions(dim)
    whiteImg.SetSpacing(*tuple(spacing))
    whiteImg.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
    whiteImg.SetOrigin(origin)

    np_arry = np.full([np.prod(dim)], 255, dtype=np.uint8)
    vtk_array = numpy_support.numpy_to_vtk(np_arry, vtk.VTK_UNSIGNED_CHAR)
    whiteImg.GetPointData().SetScalars(vtk_array)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(*tuple(spacing))
    pol2stenc.SetOutputWholeExtent(whiteImg.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImg)
    imgstenc.SetStencilData(pol2stenc.GetOutput())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    img = imgstenc.GetOutput()
    temp_vox = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    vox = temp_vox.reshape(dim[::-1])


    ctr = (bounds[:, 0] + bounds[:, 1])/2
    ctr = ctr / spacing
    ctr = ctr[::-1]
    # ctr = (ctr[3:] + ctr[:3])/2

    if return_center or return_origin or return_bounds:
        outs = [vox]
        if return_center:
            outs.append(ctr)
        if return_origin:
            voxel_origin = (-min_bound) / spacing
            outs.append(voxel_origin)
        if return_bounds:
            outs.append(bounds)
        return outs

    return vox


def polydata2voxelization(polydata, spacing, return_center=True, return_origin=False, expand=1.0):
    """
    :param polydata: vtkPolydata
    :param spacing: tuple or list of 3 (float)
    :return: voxel data and voxel_center
    """
    spacing = np.asarray(spacing)

    whiteImg = vtk.vtkImageData()
    actual_bounds = np.array(polydata.GetBounds()).reshape([-1, 2]).T

    # expand bounds, to protect voxel boundary truncation.
    bmin, bmax = actual_bounds[0], actual_bounds[1]
    ctr = (bmax + bmin) / 2
    ext = (bmax - bmin) / 2
    fext = ext * expand
    emin, emax = ctr - fext, ctr + fext
    # (3, 2)
    bounds = np.stack([emin, emax], axis=-1)

    # numpy_support.vtk_to_numpy(polydata.Get)

    dim = np.ceil((bounds[:, 1] - bounds[:, 0]) / np.array(spacing)).astype(np.int)
    ones = np.ones([3], dtype=np.int)
    dim = dim + ones
    # dim = np.ceil((bounds[:, 1] - bounds[:, 0]) / np.array(spacing)).astype(np.int)
    min_bound = bounds[:, 0]
    origin = bounds[:, 0]
    whiteImg.SetDimensions(dim)
    whiteImg.SetSpacing(*tuple(spacing))
    whiteImg.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
    whiteImg.SetOrigin(origin)

    np_arry = np.full([np.prod(dim)], 255, dtype=np.uint8)
    vtk_array = numpy_support.numpy_to_vtk(np_arry, vtk.VTK_UNSIGNED_CHAR)
    whiteImg.GetPointData().SetScalars(vtk_array)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(*tuple(spacing))
    pol2stenc.SetOutputWholeExtent(whiteImg.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImg)
    imgstenc.SetStencilData(pol2stenc.GetOutput())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    img = imgstenc.GetOutput()
    temp_vox = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    vox = temp_vox.reshape(dim[::-1])

    ctr = (bounds[:, 0] + bounds[:, 1])/2
    ctr = ctr / spacing
    ctr = ctr[::-1]

    if return_center or return_origin:
        outs = [vox]
        if return_center:
            outs.append(ctr)
        if return_origin:
            voxel_origin = (-min_bound) / spacing
            outs.append(voxel_origin)
        return outs

    return vox


def create_plane_actors(normal, origin, scale, invert=False):
    if invert:
        normal = normal[::-1]
        origin = origin[::-1]
    plane = vtk.vtkPlaneSource()

    # plane.SetCenter(*plane_normal[::-1])
    plane.SetNormal(*normal)
    plane.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(plane.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.GetProperty().SetOpacity(0.2)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(5)
    actor.GetProperty().LightingOff()

    transform = myTransform()
    transform.Translate(*tuple(origin))
    transform.Scale(scale, scale, scale)
    actor.SetUserTransform(transform)
    return actor


def show_actors(actors):

    ren = vtk.vtkRenderer()
    ctrs = []
    for act in actors:
        if isinstance(act, vtk.vtkVolume):
            ren.AddVolume(act)
        elif isinstance(act, vtk.vtkActor):
            ren.AddActor(act)
        else:
            ren.AddActor(act)
        if hasattr(act, "GetCenter"):
            ctrs.append(act.GetCenter())

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(500, 500)

    ctrs = np.mean(ctrs, axis=0)
    # ren.GetActiveCamera()->SetViewUp(0, 1, 0);
    ren.GetActiveCamera().SetFocalPoint(*tuple(ctrs))
    # ren.GetActiveCamera()->SetPosition(c[0], c[1], c[2]);

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.AddObserver("KeyPressEvent", keypress_event)
    iren.Initialize()

    renWin.Render()
    iren.Start()


def polydata2actor(polydata):
    volumeMapper = vtk.vtkPolyDataMapper()
    volumeMapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(volumeMapper)
    return actor

# def show_actors(actors):

def show(polydata):
    volumeMapper = vtk.vtkPolyDataMapper()
    volumeMapper.SetInputData(polydata)
    # volumeMapper.SetBlendModeToComposite()

    actor = vtk.vtkActor()
    actor.SetMapper(volumeMapper)

    ren = vtk.vtkRenderer()
    ren.AddVolume(actor)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    ren.GetActiveCamera().SetFocalPoint(*actor.GetCenter())

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()


def convert_voxel_to_polydata(vtkimg, threshold, radius=1., stddev=2.):
    if isinstance(vtkimg, np.ndarray):
        vtkimg = convert_numpy_vtkimag(vtkimg)
    boneExtractor = vtk.vtkMarchingCubes()
    gaussianRadius = radius
    gaussianStandardDeviation = stddev
    gaussian = vtk.vtkImageGaussianSmooth()
    gaussian.SetStandardDeviations(gaussianStandardDeviation, gaussianStandardDeviation, gaussianStandardDeviation)
    gaussian.SetRadiusFactors(gaussianRadius, gaussianRadius, gaussianRadius)

    gaussian.SetInputData(vtkimg)

    boneExtractor.SetInputConnection(gaussian.GetOutputPort())
    boneExtractor.SetValue(0, threshold)
    boneExtractor.Update()

    return boneExtractor.GetOutput()


def smoothingPolydata(polydata, iterration=15, factor=0.6, edge_smoothing=False):
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(polydata)
    smoothFilter.SetNumberOfIterations(iterration)
    smoothFilter.SetRelaxationFactor(factor)
    if not edge_smoothing:
        smoothFilter.FeatureEdgeSmoothingOff()
    else:
        smoothFilter.FeatureEdgeSmoothingOn()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()
    return smoothFilter.GetOutput()


class myTransform(vtk.vtkTransform):
    def __init__(self, ndarray=None):
        super(myTransform, self).__init__()

        if isinstance(ndarray, np.ndarray) and ndarray.shape == (4, 4):
            self.set_from_numpy_mat(ndarray)

    def getRigidTransform(self):
        out = myTransform()
        out.Translate(*self.GetPosition())
        out.RotateWXYZ(*self.GetOrientationWXYZ())
        return out

    def convert_np_mat(self):
        mat = self.GetMatrix()
        np_mat = np.zeros([4, 4], dtype=np.float64)
        for i in range(4):
            for j in range(4):
                np_mat[i, j] = mat.GetElement(i, j)
        return np_mat

    def GetInverse(self, vtkMatrix4x4=None):
        # inverse_t = super(myTransform, self).GetInverse()
        mat4x4 = self.convert_np_mat()
        t = myTransform()
        t.set_from_numpy_mat(np.linalg.inv(mat4x4))
        return t

    def set_from_numpy_rotatewxyz(self, orientWXYZ=None, trans=None, scales=None):
        if orientWXYZ is not None:
            self.RotateWXYZ(*tuple(orientWXYZ))

        if trans is not None:
            self.Translate(*tuple(trans))

        if scales is not None:
            self.Scale(scales, scales, scales)

    def set_from_numpy_mat(self, np_mat, invert=False):
        if invert:
            invmat = np.eye(4)
            invmat[:3, :3] = np_mat[:3, :3][::-1, ::-1]
            invmat[:3, 3] = np_mat[:3, 3][::-1]
            np_mat = invmat

        mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                mat.SetElement(i, j, np_mat[i, j])
        self.SetMatrix(mat)

    def set_from_numpy(self, orient=None, trans=None, scales=None):
        """
        :param orient: 3 array
        :param trans:  3 array
        :param scales: 3 array
        :return:
        """
        if orient is not None:
            self.RotateZ(orient[0])
            self.RotateX(orient[1])
            self.RotateY(orient[2])

        if trans is not None:
            self.Translate(*tuple(trans))

        if scales is not None:
            self.Scale(*tuple(scales))


    def transfrom_numpy(self, np_pts):
        np_mat = self.convert_np_mat()
        ex_pts = np.ones([np_pts.shape[0], 4])
        ex_pts[:, :3] = np_pts
        out = np.dot(np_mat, ex_pts.T).T[:, :3]
        return out

    def transform_only_rotate(self, np_pts):
        np_mat = self.convert_np_mat()
        np_mat[:, 3] = np.array([0, 0, 0, 1])
        ex_pts = np.ones([np_pts.shape[0], 4])
        ex_pts[:, :3] = np_pts
        out = np.dot(np_mat, ex_pts.T).T[:, :3]
        return out

def reconstruct_polydata(points, polys):
    vtk_points_array = numpy_support.numpy_to_vtk(points, vtk.VTK_FLOAT)
    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(vtk_points_array)

    cell_array = numpy_support.numpy_to_vtk(polys.reshape([-1, 4]), array_type=vtk.VTK_ID_TYPE)

    cells = vtk.vtkCellArray()
    polys_reshape = polys.reshape([-1, 4])
    cells.SetCells(polys_reshape.shape[0], cell_array)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(vtkpoints)
    polyData.SetPolys(cells)

    return polyData

def get_vert_indices(polydata: vtk.vtkPolyData, points):
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(polydata)
    loc.BuildLocator()
    inds = []
    for pt in points:
        i = loc.FindClosestPoint(*pt)
        inds.append(i)
    return np.asarray(inds)

# numpy converting

def vtk_2_vf(polydata_or_actor):
    if issubclass(type(polydata_or_actor), vtk.vtkActor):
        pts, polys = _actor_2_vf(polydata_or_actor)
    elif issubclass(type(polydata_or_actor), vtk.vtkPolyData):
        pts = numpy_support.vtk_to_numpy(polydata_or_actor.GetPoints().GetData())
        polys = numpy_support.vtk_to_numpy(polydata_or_actor.GetPolys().GetData())
    else:
        raise ValueError
    return pts, polys.reshape([-1, 4])


def _actor_2_vf(act):
    v = _actor_2_numpy(act)
    f = _actor_2_numpy_polys(act)
    return v, f


def _actor_2_numpy(act):
    return numpy_support.vtk_to_numpy(act.GetMapper().GetInput().GetPoints().GetData())


def _actor_2_numpy_polys(act):
    return numpy_support.vtk_to_numpy(act.GetMapper().GetInput().GetPolys().GetData()).reshape([-1, 4])


def _get_normal_property(threshold=140, max_value=255, division=3, opacity=0.8, colors=None):
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)

    color = vtk.vtkColorTransferFunction()
    if colors is None:
        colors_list = [antique_white, chartreuse, blue_light, carrot]
    else:
        colors_list = [colors] * 5

    values = np.linspace(threshold, max_value, division)
    for v, c in zip(values, colors_list):
        color.AddRGBPoint(v, *c)

    color.AddRGBPoint(threshold, *colors_list[0])

    volumeProperty.SetColor(color)
    volumeProperty.SetScalarOpacity(_get_opacity_property(threshold, opacity))
    return volumeProperty
