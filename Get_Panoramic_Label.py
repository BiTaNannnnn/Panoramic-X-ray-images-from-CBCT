##############################
# generate px image to label
# MA Wen 2023.3.9
##############################
# important
from preprocess import *
import cv2
import open3d as o3d
from sympy import Point, Ray, Circle
import trimesh
import copy
import cv2 as cv
from trimesh.voxel import creation
import matplotlib.pyplot as plt
from utils.local_io import *
from preprocess.case_op import *
from utils.show import *
from preprocess.arch_mask import *
from preprocess.skeleton import *
from preprocess.MPR import *
from tqdm import tqdm
from preprocess.projection import *

# read and save nii files, and pre-process 3d image
# if MODE = test, the intermediate result will be saved

data_root = 'DATA_500'
# idea px = 最初得到的px
ideal_px_dir = check_dir(join_path(data_root, 'ideal_px'))
# final px = 512 *512 的px
final_px_dir = check_dir(join_path(data_root, 'final_px512'))
tmp_bone_dir = check_dir(join_path(data_root, 'bone_nii'))
mat_dir = check_dir(join_path(data_root, 'mat'))

##################################
#  读取一下原始cbct的nrrd 获得里面的space信息
##################################


def read_nrrd(case_id):
    # load nrrd
    case_id = case_id
    root_path = '/data/mawen/DATA/Seg_Teeth_nrrd/teeth/'
    case_path = root_path + case_id + '_teeth_1022.seg.nrrd'
    print("\n case_path = " + case_path)
    options = nrrd.read(case_path)
    # get spacing of nrrd
    spacing = options[1]['space directions']
    x = spacing[0][0]
    y = spacing[1][1]
    z = spacing[2][2]
    translation = np.array([x, y, z])
    print(translation)
    # shape of nrrd
    shape_nrrd = options[0].shape

    return x, shape_nrrd


def Target_Transform(case_id, target_path, transformation):
    target_up_mesh = o3d.io.read_triangle_mesh(target_path)
    # mesh 2 pcd
    # target_up_mesh = target_up_mesh.sample_points_uniformly(number_of_points=200000)
    target_up_mesh = copy.deepcopy(target_up_mesh)
    print('transformation = ', transformation)
    transformation_ni = np.linalg.inv(transformation)
    print('transformation_ni = ', transformation_ni)
    # 逆矩阵
    target_up_mesh = target_up_mesh.transform(transformation_ni)
    target_up_mesh = target_up_mesh.paint_uniform_color([1, 0, 1])  # pink
    # 1/spacing 转换成原始1.0大小 与 nrrd 数据重合
    scale_nrrd, shape_nrrd = read_nrrd(case_id)
    scale_factor = 1 / scale_nrrd
    print(scale_factor)
    target_up_mesh.scale(scale_factor, center=(0, 0, 0))
    # o3d.io.write_triangle_mesh('/data/mawen/Project/Target_mesh_Up.ply', target_up_mesh)

    return target_up_mesh


def get_toothe_array(case_id):
    # load pcd
    root_path_target = '/data/mawen/DATA/CBCT_data_2021_8_13_Processed/'
    target_path = root_path_target + case_id +'/Up_Root.stl'
    root_path_source = '/data/mawen/DATA/Seg_Teeth_Mesh_Scale/'
    # single_source_path = root_path_source + case_id + '/all_Root.stl'
    single_source_path1 = '/data/mawen/DATA/Seg_Teeth_Mesh_Scale_test/C01005835067/all_Root_1.0.stl'
    trans_txt_path = root_path_source + case_id + '/Registration_matrix_Up.txt'
    transformation = np.loadtxt(trans_txt_path, delimiter=',')
    target_up_mesh = Target_Transform(case_id, target_path, transformation)
    o3d.io.write_triangle_mesh('/data/mawen/Project/target_up_mesh_2source.ply', target_up_mesh)

    # # test result 1.0scale
    # source_temp = o3d.io.read_triangle_mesh(single_source_path1)
    # source_temp = copy.deepcopy(source_temp)
    # # mesh to pcd
    # # source_temp = source_temp.sample_points_uniformly(number_of_points=200000)
    # source_temp = source_temp.paint_uniform_color([1, 1, 0])  # yellow
    #
    # # test： target 和原始nrrd数据重合
    # result = target_up_mesh + source_temp
    # o3d.io.write_triangle_mesh('/data/mawen/Project/up_transform_test.ply', result)


    # trimesh体素化
    voxel_grid = trimesh.load('/data/mawen/Project/target_up_mesh_2source.ply')
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
    box_y = round(bounding_box_min[1])
    box_z = round(bounding_box_min[2])

    # Get bounding box end 坐标
    box_end_x = box_x + voxel_shape0
    box_end_y = box_y + voxel_shape1
    box_end_z = box_z + voxel_shape2

    teeth_array = np.zeros((cbct_shape0, cbct_shape1, cbct_shape2), dtype='int')
    teeth_array[box_x:box_end_x, box_y:box_end_y, box_z:box_end_z] = voxel_grid
    # teeth_array[296:515, 145:302, 232:342] = voxel_grid

    return teeth_array


# generate panoramic x-ray image important!
def prepare_mat(data_id, nii_path, MODE):
    # load nii
    case_id = data_id
    case_name = data_id
    # case_path = '/data/mawen/DATA/nii_gz/nii_raw/C01005960842.nii.gz'
    case_path = join_path(nii_path, data_id + '.nii.gz')
    print("\n case_name = " + case_name)
    cbct_image, affine_matrix = read_nii(case_path, AFFINE=True)


    # Step 1: get MIP image #3D input
    axial_slices = get_axial_slices(cbct_image)
    axial_mip = generate_MIP(axial_slices, direction='axial')
    if MODE == 'test':
        show_rotate(axial_mip, 'Axial MIP')


    # Step 2: get arch mask
    arch_mask = get_toothe_array(case_id)
    arch_mask = generate_MIP(arch_mask, direction='axial')
    if MODE == 'test':
        show_rotate(arch_mask, 'Axial Mask')
    arch_mask = smooth_mask(arch_mask, filt_it = 1)
    if MODE == 'test':
        show_rotate(arch_mask, 'Axial MIP')


    # Step 3: get arch skeleton
    skeleton_image = get_skeleton(arch_mask)
    if MODE == 'test':
        show_rotate(skeleton_image, 'skeleton')

    # Step 4: get curve of the skeleton
    curve, ends, keypoints = get_curve(skeleton_image)

    if MODE == 'test':
        show_curve(curve, ends, ideal_px_dir, case_name, skeleton_image, title='Projection Curve')

    # Step 5, get sample bounds
    sample_points = get_sample_points(curve, ends, sample_n=288*2)
    # arch_thickness = get_arch_thichness(arch_mask, sample_points, scalar=1.5)
    arch_thickness = 40
    sample_bounds = get_border_points(sample_points, arch_thickness)
    # show upper boundary and lower boundary
    if MODE == 'test':
        show_sample_bounds(axial_mip, sample_bounds, title='Axial Curve')

    # Step 7: generate bone and MPR images
    cbct_copy = cbct_image.copy()
    bone_mask = cbct_copy > 1500
    save_nii(bone_mask.astype(np.uint8), affine_matrix, case_name+'.nii.gz', tmp_bone_dir)

    # 下面这句mask 很多信息 全景片比较黑 舍去
    # cbct_image[bone_mask == 0] = 0
    MPR_images = get_MPR_images(cbct_image, sample_bounds, arch_thickness)

    px_img_ideal = assemble_MPR_images(MPR_images)
    px_img_ideal = norm_px(px_img_ideal)
    px_img_ideal = px2img(px_img_ideal)

    show_result(px_img_ideal, 'Ideal Panoramic Image')
    save_png(px_img_ideal, ideal_px_dir, case_name + '.png')
    img_path = join_path(ideal_px_dir, case_name + '.png')
    px_img_ideal = cv2.imread(img_path)
    px_img_ideal = cv2.flip(px_img_ideal, 0)
    px_img_ideal = cv2.resize(px_img_ideal, (512, 512))
    save_png(px_img_ideal, final_px_dir, case_name + '.png')
    print(px_img_ideal.shape)

    if MODE == 'test':
        show_result(px_img_ideal, 'Ideal Panoramic Image')

    if MODE == 'run':
        mat = { 'CBCT': np.array(cbct_image, dtype=np.float32),
                'Bone': np.array(bone_mask, dtype=np.uint8),
                'MPR': np.array(MPR_images, dtype=np.float32),
                'Case_ID': case_id,
                'PriorShape': np.array(sample_points, dtype=np.float32),
                'Ideal_PX': np.array(px_img_ideal, dtype=np.float32),
                }
        save_mat(mat, case_name + '.mat', mat_dir)


# if mode is test, intermediate result will be shown
if __name__ == '__main__':
    nii_path = '/data/mawen/DATA/nii_gz/nii_raw'  # 更改路径
    # myparth = '/data/mawen/DATA/nii_gz/nii_raw/'
    files = os.listdir(nii_path)
    total_n = len(files)
    # MODE = 'run'
    MODE = 'test'

    for file in files:
        data_id = file.split('.', 3)[0]
        jpg = str('/data/mawen/Project/3D_Dental_Master/DATA_500/final_px512/' + data_id + ".png")

        # 判断图片是否存在
        if os.path.exists(jpg):
            continue
        else:
            data_id = 'C01005835067'
            prepare_mat(data_id, nii_path, MODE)
            # if data_id >= 133:
            #    break
            print("break up ! PatientID = ", data_id)