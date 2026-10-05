'''
##################################
## Get target cbct label
## 把target * 逆矩阵 /spacing = source 对应的牙齿
## --> 得到label 的 teeth array --> 得到有label的全景片 mask
   具体赋值规则：Impacted_101 ====》 11  Impacted牙齿去掉中间的0 ：201= =21 301 = 31
               11 ==== 11*5 === 55  其余牙齿都*5
##
##  important
'''


import os.path
import copy
import cv2
from preprocess.skeleton import *
from preprocess.MPR import *
from setuptools import glob
import open3d as o3d
import trimesh
from trimesh.voxel import creation
from utils.local_io import *
from preprocess.arch_mask import *
from preprocess.projection import *
import natsort
import trimesh.voxel.ops as tvo

data_root = 'xxx/Project/3D_Dental_Master/DATA_SET'
label_px_dir = check_dir(join_path(data_root, 'temp_label_px'))
final_label_dir = check_dir(join_path(data_root, 'final_label_px_MeshScale'))

# read and save nii files, and pre-process 3d image
# if MODE = test, the intermediate result will be saved

# data_root = 'DATA_SET'
# idea px = 最初得到的px
# skeleton_image_dir = check_dir(join_path(data_root, 'skeleton_image'))


##################################
#  读取一下原始cbct的nrrd 获得里面的space信息
##################################

def read_nrrd(case_id):
    # load nrrd
    case_id = case_id
    root_path = 'xxx/DATA/Seg_Teeth_nrrd/teeth/'
    case_path = root_path + case_id + '_teeth_1022.seg.nrrd'
    print("\n case_path = " + case_path)
    options = nrrd.read(case_path)
    # get spacing of nrrd
    spacing = options[1]['space directions']
    x = spacing[0][0]
    y = spacing[1][1]
    z = spacing[2][2]
    translation = np.array([x, y, z])
    # print(translation)

    # shape of nrrd
    shape_nrrd = options[0].shape

    return x, shape_nrrd

##################################
#  location.txt变换到全局坐标系
##################################
def location_traget(teeth_folder_path, idx=0, url=None):
    # teeth_path_dict = {int(os.path.basename(path).split('.')[0]): path for path in
    #                    glob.glob(os.path.join(teeth_folder_path, f'*._Crown.stl'))}
    # print(glob.glob(os.path.join(teeth_folder_path, 'Location.txt')))
    step_files = natsort.natsorted(glob.glob(os.path.join(teeth_folder_path, 'Location.txt')))
    transformation_matrix_dict = {}
    for line in np.loadtxt(step_files[idx]):
        tid = int(line[0])

        translation_matrix = trimesh.transformations.translation_matrix(line[1:4])
        rotation_matrix = trimesh.transformations.quaternion_matrix(np.hstack([line[-1:], line[4:-1]]))

        transformation_matrix_dict[tid] = (translation_matrix, rotation_matrix)
        # print(transformation_matrix_dict)
    return transformation_matrix_dict

def mesh_to_voxel(single_up_mesh):
    # Get bounding box 顶点
    bounding_box_min = single_up_mesh.bounding_box.bounds[0]
    bounding_box_max = single_up_mesh.bounding_box.bounds[1]
    voxel_grid = creation.voxelize(single_up_mesh, 1)
    # 密集矩阵
    voxel_grid = voxel_grid.matrix
    voxel_grid_unique = np.unique(voxel_grid)
    # print(voxel_grid_unique)
    return voxel_grid, bounding_box_min

def voxel_to_mesh(single_up_voxel, shape_nrrd, bounding_box_min):
    # 读取cbct.shape
    # print('cbct shape = ', shape_nrrd)
    cbct_shape0 = shape_nrrd[0]
    cbct_shape1 = shape_nrrd[1]
    cbct_shape2 = shape_nrrd[2]

    # size of voxel 长宽高
    voxel_shape0 = round(single_up_voxel.shape[0])
    voxel_shape1 = round(single_up_voxel.shape[1])
    voxel_shape2 = round(single_up_voxel.shape[2])
    # print('voxel_grid.shape = ', single_up_voxel.shape)

    # Get bounding box 顶点坐标
    box_x = round(bounding_box_min[0])
    box_x = np.maximum(box_x, 0)       # 将小于0的值替换为0
    box_y = round(bounding_box_min[1])
    box_y = np.maximum(box_y, 0)
    box_z = round(bounding_box_min[2])
    box_z = np.maximum(box_z, 0)
    # print(box_x, box_y, box_z)

    # Get bounding box end 坐标
    box_end_x = box_x + voxel_shape0
    box_end_y = box_y + voxel_shape1
    box_end_z = box_z + voxel_shape2
    # 判断边界溢出
    if box_end_x > cbct_shape0:
        box_end_x = cbct_shape0
        box_x = cbct_shape0 - voxel_shape0
    if box_end_y > cbct_shape1:
        box_end_y = cbct_shape1
        box_y = cbct_shape1 - voxel_shape1
    if box_end_z > cbct_shape2:
        box_end_z = cbct_shape2
        box_z = cbct_shape2 - voxel_shape2

    teeth_array = np.zeros((cbct_shape0, cbct_shape1, cbct_shape2), dtype='int')
    teeth_array[box_x:box_end_x, box_y:box_end_y, box_z:box_end_z] = single_up_voxel

    return teeth_array


def route_single_teeth(mesh_up_initial_path, teeth_number, transformation_matrix_dict):
    mesh = trimesh.load(mesh_up_initial_path)
    translation_matrix = transformation_matrix_dict[teeth_number][0]
    rotation_matrix = transformation_matrix_dict[teeth_number][1]
    mesh = mesh.apply_transform(rotation_matrix)
    single_up_mesh = mesh.apply_transform(translation_matrix)

    return single_up_mesh

def voxel_evaluation_Impacted_up(single_up_mesh, teeth_number,  scale_factor, Matrix_Up):
    single_up_mesh.export('xxx/Project/Target_single_Up.ply')
    # open3d 读入
    single_up_mesh = o3d.io.read_triangle_mesh('xxx/Project/Target_single_Up.ply')
    # mesh * 逆矩阵
    single_up_mesh = single_up_mesh.transform(Matrix_Up)
    o3d.io.write_triangle_mesh('xxx/Project/transform.ply', single_up_mesh)
    # 1/spacing 转换成原始1.0大小 与 nrrd 数据重合
    single_up_mesh.scale(scale_factor, center=(0, 0, 0))
    single_up_mesh.paint_uniform_color([1, 0, 1])  # pink
    o3d.io.write_triangle_mesh('xxx/Project/scale_up.ply', single_up_mesh)
    single_up_mesh = trimesh.load('xxx/Project/scale_up.ply')
    # mesh to voxel # 会有平移的问题
    voxel, bounding_box_min = mesh_to_voxel(single_up_mesh)
    # voxel 赋值
    voxel = np.array(voxel).astype(int)
    for i in range(len(voxel)):
        for j in range(len(voxel[i])):
            for k in range(len(voxel[i][j])):
                # 检查元素是否为True
                if voxel[i][j][k] == 1:
                    # 将True赋值为 201 ==> 21
                    a = teeth_number // 100
                    b = teeth_number % 10
                    number = (a * 10 + b)
                    voxel[i][j][k] = number
                    # voxel[i][j][k] = teeth_number * 5
    voxel_grid_unique = np.unique(voxel)
    # print(voxel_grid_unique)

    return voxel, bounding_box_min

def voxel_evaluation_Impacted_down(single_down_mesh, teeth_number,  scale_factor, Matrix_Down):
    single_down_mesh.export('xxx/Project/Target_single_Down.ply')
    # open3d 读入
    single_down_mesh = o3d.io.read_triangle_mesh('xxx/Project/Target_single_Down.ply')
    # mesh * 逆矩阵
    single_down_mesh = single_down_mesh.transform(Matrix_Down)
    o3d.io.write_triangle_mesh('xxx/Project/transform.ply', single_down_mesh)
    # 1/spacing 转换成原始1.0大小 与 nrrd 数据重合
    single_down_mesh.scale(scale_factor, center=(0, 0, 0))
    single_down_mesh.paint_uniform_color([1, 0, 1])  # pink
    o3d.io.write_triangle_mesh('xxx/Project/scale_down.ply', single_down_mesh)
    single_down_mesh = trimesh.load('xxx/Project/scale_down.ply')
    # mesh to voxel # 会有平移的问题
    voxel, bounding_box_min = mesh_to_voxel(single_down_mesh)
    # voxel 赋值
    voxel = np.array(voxel).astype(int)
    for i in range(len(voxel)):
        for j in range(len(voxel[i])):
            for k in range(len(voxel[i][j])):
                # 检查元素是否为True
                if voxel[i][j][k] == 1:
                    # 将True赋值为 201 ==> 21
                    a = teeth_number // 100
                    b = teeth_number % 10
                    number = (a * 10 + b)
                    voxel[i][j][k] = number
    voxel_grid_unique = np.unique(voxel)
    # print(voxel_grid_unique)

    return voxel, bounding_box_min


def voxel_evaluation_method_up(single_up_mesh, teeth_number,  scale_factor, Matrix_Up):
    single_up_mesh.export('xxx/Project/Target_single_Up.ply')
    # open3d 读入
    single_up_mesh = o3d.io.read_triangle_mesh('xxx/Project/Target_single_Up.ply')
    # mesh * 逆矩阵
    single_up_mesh = single_up_mesh.transform(Matrix_Up)
    o3d.io.write_triangle_mesh('xxx/Project/transform.ply', single_up_mesh)
    # 1/spacing 转换成原始1.0大小 与 nrrd 数据重合
    single_up_mesh.scale(scale_factor, center=(0, 0, 0))
    single_up_mesh.paint_uniform_color([1, 0, 1])  # pink
    o3d.io.write_triangle_mesh('xxx/Project/scale_up.ply', single_up_mesh)
    single_up_mesh = trimesh.load('xxx/Project/scale_up.ply')
    # mesh to voxel # 会有平移的问题
    voxel, bounding_box_min = mesh_to_voxel(single_up_mesh)
    # voxel 赋值
    voxel = np.array(voxel).astype(int)
    for i in range(len(voxel)):
        for j in range(len(voxel[i])):
            for k in range(len(voxel[i][j])):
                # 检查元素是否为True
                if voxel[i][j][k] == 1:
                    # 将True赋值为5
                    voxel[i][j][k] = teeth_number * 5
    voxel_grid_unique = np.unique(voxel)
    # print(voxel_grid_unique)

    return voxel, bounding_box_min
    # # test
    # source_mesh = trimesh.load('xxx/DATA/Seg_Teeth_Mesh_Scale_test/CASE_ID/all_Root_1.0.stl')
    # voxel_mesh = creation.from_compact(voxel)
    # voxel_mesh.paint_uniform_color([1, 0, 1])  # pink
    # mesh = source_mesh + voxel_mesh
    # mesh.export('xxx/Project/test.ply')



def voxel_evaluation_method_down(single_down_mesh, teeth_number,  scale_factor, Matrix_Down):
    single_down_mesh.export('xxx/Project/Target_single_Down.ply')
    # open3d 读入
    single_down_mesh = o3d.io.read_triangle_mesh('xxx/Project/Target_single_Down.ply')
    # mesh * 逆矩阵
    single_down_mesh = single_down_mesh.transform(Matrix_Down)
    o3d.io.write_triangle_mesh('xxx/Project/transform.ply', single_down_mesh)
    # 1/spacing 转换成原始1.0大小 与 nrrd 数据重合
    single_down_mesh.scale(scale_factor, center=(0, 0, 0))
    single_down_mesh.paint_uniform_color([1, 0, 1])  # pink
    o3d.io.write_triangle_mesh('xxx/Project/scale_down.ply', single_down_mesh)
    single_down_mesh = trimesh.load('xxx/Project/scale_down.ply')
    # mesh to voxel # 会有平移的问题
    voxel, bounding_box_min = mesh_to_voxel(single_down_mesh)
    # voxel 赋值
    voxel = np.array(voxel).astype(int)
    for i in range(len(voxel)):
        for j in range(len(voxel[i])):
            for k in range(len(voxel[i][j])):
                # 检查元素是否为True
                if voxel[i][j][k] == 1:
                    # 将True赋值为5
                    voxel[i][j][k] = teeth_number * 5
    voxel_grid_unique = np.unique(voxel)
    # print(voxel_grid_unique

    # # test
    # source_mesh = trimesh.load('xxx/DATA/Seg_Teeth_Mesh_Scale_test/CASE_ID/all_Root_1.0.stl')
    # voxel_mesh = creation.from_compact(voxel)
    # voxel_mesh.paint_uniform_color([1, 0, 1])  # pink
    # mesh = source_mesh + voxel_mesh
    # mesh.export('xxx/Project/test.ply')
    return voxel, bounding_box_min


def single_teeth(target_all_path, transformation_matrix_dict, Matrix_Up, Matrix_Down):
    case_id = target_all_path[-12:]
    scale_nrrd, shape_nrrd = read_nrrd(case_id)
    scale_factor = 1 / scale_nrrd
    tooth_one = os.listdir(target_all_path)
    tooth_two = os.listdir(target_all_path)
    target_up_mesh = None
    target_down_mesh = None
    for teeth in tooth_two:
        if target_down_mesh is None or target_up_mesh is None:
            # up tooth
            if teeth[0] == '1' or teeth[0] == '2':
                teeth_number = teeth.split('.', 3)[0]
                teeth_number = eval(teeth_number)
                mesh_up_initial_path = os.path.join(target_all_path, teeth)
                # route target single teeth
                single_up_mesh = route_single_teeth(mesh_up_initial_path, teeth_number, transformation_matrix_dict)
                # mesh to voxel + 赋值
                target_up_mesh, bounding_box_min = voxel_evaluation_method_up(single_up_mesh, teeth_number,
                                                           scale_factor, Matrix_Up)
                target_up_mesh = voxel_to_mesh(target_up_mesh, shape_nrrd, bounding_box_min)

            # down tooth
            elif teeth[0] == '3' or teeth[0] == '4':
                teeth_number = teeth.split('.', 3)[0]
                teeth_number = eval(teeth_number)
                mesh_up_initial_path = os.path.join(target_all_path, teeth)
                # route target single teeth
                single_down_mesh = route_single_teeth(mesh_up_initial_path, teeth_number, transformation_matrix_dict)
                # mesh to voxel + 赋值
                target_down_mesh, bounding_box_min = voxel_evaluation_method_down(single_down_mesh, teeth_number,
                                                           scale_factor, Matrix_Down)
                target_down_mesh = voxel_to_mesh(target_down_mesh, shape_nrrd, bounding_box_min)
                # mesh = trimesh.load(mesh_up_initial_path)
                # rotation_matrix = transformation_matrix_dict[teeth_number][1]
                # translation_matrix = transformation_matrix_dict[teeth_number][0]
                # mesh = mesh.apply_transform(rotation_matrix)
                # mesh = mesh.apply_transform(translation_matrix)
        else:
            break
    # print('target_up_mesh = ', target_up_mesh)
    # print('target_down_mesh = ', target_down_mesh)

    for one in tooth_one:
        # print('patient id = ', target_all_path[-12:])
        # print(one[0])
        if one[0] == 'L' or one[0] == 'U':
            # print(one)
            continue
        # up tooth
        elif one[0] == '1' or one[0] == '2' or one.split('_', 2)[1][0] == '1' or one.split('_', 2)[1][0] == '2':
            mesh_dir = os.path.join(target_all_path, one)
            teeth_number = one.split('.', 3)[0]
            if teeth_number[0] == 'I':
                teeth_number = teeth_number.split('_', 2)[1]
                teeth_number = eval(teeth_number)
                mesh = trimesh.load(mesh_dir)
                single_up_voxel, bounding_box_min = voxel_evaluation_Impacted_up(mesh, teeth_number,
                                                            scale_factor, Matrix_Up)
                # voxel to mesh
                single_up_voxel = voxel_to_mesh(single_up_voxel, shape_nrrd, bounding_box_min)
                # a[b > 0] = b[b > 0]
                target_up_mesh[single_up_voxel > 0] = single_up_voxel[single_up_voxel > 0]
                # target_up_mesh = target_up_mesh + single_up_voxel

                # test
                voxel_grid_unique = np.unique(target_up_mesh)
                # print('target_up_mesh = ', voxel_grid_unique)
            else:
                teeth_number = eval(teeth_number)
                # print(teeth_number)
                # print(mesh_dir)
                # route target single teeth
                single_up_mesh = route_single_teeth(mesh_dir, teeth_number, transformation_matrix_dict)
                # mesh to voxel + 赋值
                single_up_voxel, bounding_box_min = voxel_evaluation_method_up(single_up_mesh, teeth_number,
                                                            scale_factor, Matrix_Up)
                # voxel to mesh
                single_up_voxel = voxel_to_mesh(single_up_voxel, shape_nrrd, bounding_box_min)
                target_up_mesh[single_up_voxel > 0] = single_up_voxel[single_up_voxel > 0]
                # target_up_mesh = target_up_mesh + single_up_voxel

                # test
                voxel_grid_unique = np.unique(target_up_mesh)
                # print('target_up_mesh = ', voxel_grid_unique)

        # down tooth
        elif one[0] == '3' or one[0] == '4' or one.split('_', 2)[1][0] == '3' or one.split('_', 2)[1][0] == '4':
            mesh_dir = os.path.join(target_all_path, one)
            teeth_number = one.split('.', 3)[0]
            if teeth_number[0] == 'I':
                teeth_number = teeth_number.split('_', 2)[1]
                teeth_number = eval(teeth_number)
                mesh = trimesh.load(mesh_dir)
                single_down_voxel, bounding_box_min = voxel_evaluation_Impacted_down(mesh, teeth_number,
                                                            scale_factor, Matrix_Down)
                # voxel to mesh
                single_down_voxel = voxel_to_mesh(single_down_voxel, shape_nrrd, bounding_box_min)
                target_down_mesh[single_down_voxel > 0] = single_down_voxel[single_down_voxel > 0]
                # target_down_mesh = target_down_mesh + single_down_voxel

                # test
                voxel_grid_unique = np.unique(target_down_mesh)
                # print('target_down_mesh = ', voxel_grid_unique)

            else:
                teeth_number = eval(teeth_number)
                # print(teeth_number)
                # print(mesh_dir)
                # route target single teeth
                single_down_mesh = route_single_teeth(mesh_dir, teeth_number, transformation_matrix_dict)
                # mesh to voxel + 赋值
                single_down_voxel, bounding_box_min = voxel_evaluation_method_up(single_down_mesh, teeth_number,
                                                            scale_factor, Matrix_Down)
                # voxel to mesh
                single_down_voxel = voxel_to_mesh(single_down_voxel, shape_nrrd, bounding_box_min)
                target_down_mesh[single_down_voxel > 0] = single_down_voxel[single_down_voxel > 0]
                # target_down_mesh = target_down_mesh + single_down_voxel
                voxel_grid_unique = np.unique(target_down_mesh)
                # print('target_down_mesh = ', voxel_grid_unique)
        else:
            # print(one)
            continue
    # tooth_voxel_all = target_down_mesh + target_up_mesh
    target_down_mesh[target_up_mesh > 0] = target_up_mesh[target_up_mesh > 0]
    tooth_voxel_all = target_down_mesh
    voxel_grid_unique = np.unique(tooth_voxel_all)
    # print('target_down_mesh = ', voxel_grid_unique)
    # # test to mesh show
    tooth_mesh_all = tvo.matrix_to_marching_cubes(tooth_voxel_all)
    tooth_mesh_all = trimesh.Trimesh(vertices=tooth_mesh_all.vertices, faces=tooth_mesh_all.faces)
    tooth_mesh_all.visual = trimesh.visual.color.ColorVisuals(vertex_colors=None, face_colors=(1, 0, 1, 1))

    tooth_mesh_all.export('xxx/Project/all_tooth_voxel2mesh.ply')
    return tooth_voxel_all, tooth_mesh_all


def get_transformation(Matrix_Up, Matrix_Down):
    # load  transform  txt
    Matrix_Up = np.loadtxt(Matrix_Up, delimiter=',')
    Matrix_Down = np.loadtxt(Matrix_Down, delimiter=',')
    # get up 逆矩阵
    # print('Matrix_Up = ', Matrix_Up)
    transformation_up_ni = np.linalg.inv(Matrix_Up)
    # print('transformation_up_ni = ', transformation_up_ni)

    # get down 逆矩阵
    transformation_down_ni = np.linalg.inv(Matrix_Down)
    # print('transformation_down_ni = ', transformation_down_ni)

    return transformation_up_ni, transformation_down_ni



def get_label_panoramic(case_id, nii_path, MODE, curve, tooth_voxel_all):
    # load nii
    data_id = case_id
    # case_path = 'xxx/DATA/nii_gz/nii_raw/CASE_ID.nii.gz'
    case_path = join_path(nii_path, data_id + '.nii.gz')
    print("\n case_name = " + data_id)
    cbct_image, affine_matrix = read_nii(case_path, AFFINE=True)

    # Step 1: get MIP image #3D input
    # axial_slices = get_axial_slices(cbct_image, MODE)
    axial_mip = generate_MIP(cbct_image, direction='axial')
    if MODE == 'test':
        show_rotate(axial_mip, 'Axial MIP')

    # Step 2: get arch mask
    arch_mask = get_toothe_array(case_id)
    arch_mask = generate_MIP(arch_mask, direction='axial')
    if MODE == 'test':
        show_rotate(arch_mask, 'Axial Mask')
    arch_mask = smooth_mask(arch_mask, filt_it=1)
    if MODE == 'test':
        show_rotate(arch_mask, 'Axial MIP')
    #
    # # Step 3: get arch skeleton
    # skeleton_image = get_skeleton(arch_mask)
    # save_path = join_path(skeleton_image_dir, data_id + '_skeleton.txt')
    # np.savetxt(save_path, skeleton_image, fmt='%f', delimiter=',')
    # # # 检查保存txt结果
    # # b = np.loadtxt(save_path, delimiter=',')
    # # print(b)
    skeleton_image = curve
    if MODE == 'test':
        show_rotate(curve, 'skeleton image')

    # Step 4: get curve of the skeleton
    curve, ends, keypoints = get_curve(skeleton_image)

    if MODE == 'test':
        show_curve(curve, ends, label_px_dir, data_id, skeleton_image, title='Projection Curve')

    # Step 5, get sample bounds
    sample_points = get_sample_points(curve, ends, sample_n=288*2)
    arch_thickness = get_arch_thichness(arch_mask, sample_points, scalar=1.5)
    # arch_thickness = 40
    sample_bounds = get_border_points(sample_points, arch_thickness)
    # show upper boundary and lower boundary
    if MODE == 'test':
        show_sample_bounds(axial_mip, sample_bounds, title='Axial Curve')

    # Step 7: generate bone and MPR images
    # cbct_copy = cbct_image.copy()
    # bone_mask = cbct_copy > 1500
    # save_nii(bone_mask.astype(np.uint8), affine_matrix, data_id+'.nii.gz', tmp_bone_dir)

    # 下面这句mask 很多信息 全景片比较黑 舍去
    # cbct_image[bone_mask == 0] = 0
    # MPR_images = get_MPR_images(cbct_image, sample_bounds, arch_thickness)
    voxel_grid_unique = np.unique(tooth_voxel_all)
    print(voxel_grid_unique)
    MPR_images = get_MPR_images(tooth_voxel_all, sample_bounds, arch_thickness)
    px_img_ideal = label_MPR_images(MPR_images)
    voxel_grid_unique = np.unique(px_img_ideal)
    print(voxel_grid_unique)

    # 屏蔽 > 240 的数
    # px_img_ideal[px_img_ideal > 240] = 0
    # px_img_ideal = norm_px(px_img_ideal)
    # px_img_ideal = px2img(px_img_ideal)
    if MODE == 'test':
        show_label(px_img_ideal, 'Ideal label Panoramic Image')
    px_img_ideal = cv2.flip(px_img_ideal, 0)

    # px_img_ideal = cv2.resize(px_img_ideal, (512, 512)) # resize 差值得到的数字会改变
    # reshape nearest 得到的数值不会改变
    # save as 512*512
    # px_img_ideal = cv2.resize(px_img_ideal, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    voxel_grid_unique = np.unique(px_img_ideal)
    print('voxel_grid_unique = ', voxel_grid_unique)
    print('px_img_ideal.shape = ', px_img_ideal.shape)
    # save picture
    # px_img_ideal = np.flip(px_img_ideal, axis=0)
    final_save_path = join_path(final_label_dir, data_id + '_mask.png')
    cv2.imwrite(final_save_path, px_img_ideal)
    if MODE == 'test':
        show_label(px_img_ideal, 'Ideal label Panoramic Image finally')

    # save_png(px_img_ideal, final_label_dir, data_id + '.png')


def Target_Transform(case_id, target_path, transformation):
    target_up_mesh = o3d.io.read_triangle_mesh(target_path)
    # mesh 2 pcd
    # target_up_mesh = target_up_mesh.sample_points_uniformly(number_of_points=200000)
    target_up_mesh = copy.deepcopy(target_up_mesh)
    # print('transformation = ', transformation)
    transformation_ni = np.linalg.inv(transformation)
    # print('transformation_ni = ', transformation_ni)
    # 逆矩阵
    target_up_mesh = target_up_mesh.transform(transformation_ni)
    target_up_mesh = target_up_mesh.paint_uniform_color([1, 0, 1])  # pink
    # 1/spacing 转换成原始1.0大小 与 nrrd 数据重合
    scale_nrrd, shape_nrrd = read_nrrd(case_id)
    scale_factor = 1 / scale_nrrd
    print(scale_factor)
    target_up_mesh.scale(scale_factor, center=(0, 0, 0))
    # o3d.io.write_triangle_mesh('xxx/Project/Target_mesh_Up.ply', target_up_mesh)

    return target_up_mesh


def get_toothe_array(case_id):
    # load pcd
    root_path_target = 'xxx/DATA/CBCT_data_2021_8_13_Processed/'
    target_path = root_path_target + case_id +'/Up_Root.stl'
    root_path_source = 'xxx/DATA/Seg_Teeth_Mesh_Scale/'
    single_source_path = root_path_source + case_id + '/all_Root.stl'
    single_source_path1 = 'xxx/DATA/Seg_Teeth_Mesh_Scale_test/CASE_ID/all_Root_1.0.stl'
    trans_txt_path = root_path_source + case_id + '/Registration_matrix_Up.txt'
    transformation = np.loadtxt(trans_txt_path, delimiter=',')
    target_up_mesh = Target_Transform(case_id, target_path, transformation)
    o3d.io.write_triangle_mesh('xxx/Project/target_up_mesh_2source.ply', target_up_mesh)

    # # test result 1.0scale
    # source_temp = o3d.io.read_triangle_mesh(single_source_path1)
    # source_temp = copy.deepcopy(source_temp)
    # # mesh to pcd
    # # source_temp = source_temp.sample_points_uniformly(number_of_points=200000)
    # source_temp = source_temp.paint_uniform_color([1, 1, 0])  # yellow
    #
    # # test： target 和原始nrrd数据重合
    # result = target_up_mesh + source_temp
    # o3d.io.write_triangle_mesh('xxx/Project/up_transform_test.ply', result)


    # trimesh体素化
    voxel_grid = trimesh.load('xxx/Project/target_up_mesh_2source.ply')
    # Get bounding box 顶点
    bounding_box_min = voxel_grid.bounding_box.bounds[0]
    bounding_box_max = voxel_grid.bounding_box.bounds[1]
    voxel_grid = creation.voxelize(voxel_grid, 1)
    # print(transformation)
    # voxel_grid = voxel_grid.transform(transformation)

    # 密集矩阵
    voxel_grid = voxel_grid.matrix
    voxel_grid_unique = np.unique(voxel_grid)
    print(voxel_grid_unique)

    # get shape of nrrd 原始cbct大小
    x, shape_nrrd = read_nrrd(case_id)
    print('cbct shape = ', shape_nrrd)
    cbct_shape0 = shape_nrrd[0]
    cbct_shape1 = shape_nrrd[1]
    cbct_shape2 = shape_nrrd[2]

    # size of voxel 长宽高
    voxel_shape0 = round(voxel_grid.shape[0])
    voxel_shape1 = round(voxel_grid.shape[1])
    voxel_shape2 = round(voxel_grid.shape[2])
    print('voxel_grid.shape = ', voxel_grid.shape)

    # Get bounding box 顶点坐标
    box_x = round(bounding_box_min[0])
    box_x = np.maximum(box_x, 0)   # 将小于0的值替换为0
    box_y = round(bounding_box_min[1])
    box_y = np.maximum(box_y, 0)
    box_z = round(bounding_box_min[2])
    box_z = np.maximum(box_z, 0)
    print(box_x, box_y, box_z)


    # Get bounding box end 坐标
    box_end_x = box_x + voxel_shape0
    box_end_y = box_y + voxel_shape1
    box_end_z = box_z + voxel_shape2
    if box_end_x > cbct_shape0:
        box_end_x = cbct_shape0
        box_x = cbct_shape0 - voxel_shape0
    if box_end_y > cbct_shape1:
        box_end_y = cbct_shape1
        box_y = cbct_shape1 - voxel_shape1
    if box_end_z > cbct_shape2:
        box_end_z = cbct_shape2
        box_z = cbct_shape2 - voxel_shape2

    teeth_array = np.zeros((cbct_shape0, cbct_shape1, cbct_shape2), dtype='int')
    teeth_array[box_x:box_end_x, box_y:box_end_y, box_z:box_end_z] = voxel_grid
    # teeth_array[296:515, 145:302, 232:342] = voxel_grid

    return teeth_array




if __name__ == '__main__':
    nii_path = 'xxx/DATA/nii_row'  # 更改路径
    target_mesh_path = "xxx/DATA/CBCT_data_2021_8_13_Processed"
    # single_source_path = "xxx/DATA/Seg_Teeth_Mesh_Scale_test"
    single_source_path = "xxx/DATA/Seg_Teeth_Mesh_Scale"
    # 牙弓曲线保存地址
    skeleton_save_path = "xxx/Project/3D_Dental_Master/DATA_SET/skeleton_image/"
    MODE = 'run'
    # MODE = 'test'
    for root, dirs, files in os.walk(target_mesh_path):
        for dir in dirs:
            # dir = 'CASE_ID'
            case_id = dir

            target_id_path = os.path.join(target_mesh_path, dir)
            Matrix_Up_path = single_source_path + "/" + case_id + "/" + "Registration_matrix_Up.txt"
            Matrix_Down_path = single_source_path + "/" + case_id + "/" + "Registration_matrix_Down.txt"
            label_save_path = final_label_dir + '/' + case_id + '_mask.png'
            nrrd_path = str('xxx/DATA/Seg_Teeth_nrrd/teeth/' + case_id + '_teeth_1022.seg.nrrd')
            # 判断全景片是否存在
            if os.path.exists(label_save_path):
                continue
            elif case_id == 'CASE_ID' or case_id == 'CASE_ID' or case_id == 'CASE_ID' or case_id == 'CASE_ID':
                continue
            elif os.path.exists(nrrd_path):
                # 得到location.txt变换到全局坐标系
                transformation_matrix_dict = location_traget(target_id_path, idx=0, url=None)
                # 得到上下颌逆矩阵
                Matrix_Up, Matrix_Down = get_transformation(Matrix_Up_path, Matrix_Down_path)
                # 每颗牙齿的mesh + 设置数字
                tooth_voxel_all, tooth_mesh_all = single_teeth(target_id_path,
                                                               transformation_matrix_dict,
                                                               Matrix_Up, Matrix_Down)
                voxel_grid_unique = np.unique(tooth_voxel_all)
                # print(voxel_grid_unique)

                # save temp result
                np.save('xxx/Project/3D_Dental_Master/matrix.npy', tooth_voxel_all)

                # test
                tooth_voxel_all = np.load('xxx/Project/3D_Dental_Master/matrix.npy')

                # 得到牙弓曲线
                save_path = skeleton_save_path + case_id + '_skeleton.txt'
                curve = np.loadtxt(save_path, delimiter=',')
                print(curve.shape)
                # 得到全景片的label
                result = get_label_panoramic(case_id, nii_path, MODE, curve, tooth_voxel_all)
                print("######   Finished One ID ! PatientID = ", case_id)

            else:
                continue

    print("finished all ids ")




# CASE_ID label结果不正常
# CASE_ID ValueError: could not broadcast input array from shape (88,99,155) into shape (88,99,6)
# CASE_ID 本身数据格式有问题（xyz）
# CASE_ID / CASE_ID /CASE_ID  dcm格式有问题