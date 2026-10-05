##############################
# generate px image to label
# XXX 2024.4.19
# Get 全景片 根据牙弓曲线展开做平均值
# 得到全景片 每颗牙齿大小与真是牙齿大小一致
##############################
# important
import cv2
import open3d as o3d
import trimesh
import pickle
from trimesh.voxel import creation
from utils.local_io import *
from preprocess.case_op import *
from utils.show import *
from preprocess.arch_mask import *
from preprocess.skeleton import *
from preprocess.MPR import *
from preprocess.projection import *

# read and save nii files, and pre-process 3d image
# if MODE = test, the intermediate result will be saved

data_root = 'xxx/DATA/Panoramic_Max'
# idea px = 最初得到的px
ideal_px_dir = check_dir(join_path(data_root, 'ideal_px'))
# final px = 512 *512 的px
final_px_dir = check_dir(join_path(data_root, 'final_px512'))

pkl_dir = check_dir(join_path(data_root, 'mapping_table'))

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
    print(translation)
    # shape of nrrd
    shape_nrrd = options[0].shape

    return x, shape_nrrd


def generate_arc_points(cbct_img, num_points=512):
    # 获取 CBCT 图像的宽度和高度
    cbct_heigh, cbct_width, cbct_depth = np.shape(cbct_img)  # (yxz)

    # 原点在 CBCT 中心
    center_y = cbct_heigh // 2
    center_x = cbct_width // 2
    center_z = cbct_depth // 2
    '''3d坐标系中心'''
    center_point = (center_x, center_y, center_z)

    # 圆弧的半径为 CBCT 深度的 2/3
    diagonal_length = math.sqrt(cbct_heigh ** 2 + cbct_width ** 2)

    # 圆弧的半径为 CBCT 深度的 3/4
    radius = diagonal_length * 3 / 4

    # 生成圆弧上的点 1）2/7-5/7 show
    # start_angle = -np.pi * 3 / 7  # right
    # end_angle = np.pi * 7 / 13  # left
    start_angle = -1.7
    end_angle = 1.7
    angles = np.linspace(start_angle, end_angle, num_points)
    arc_points_x = center_x + radius * np.cos(angles)
    arc_points_y = center_y - radius * np.sin(angles)
    # 计算每个点到中心点的垂直距离
    # vertical_distances = center_y - np.sqrt(radius ** 2 - (arc_points_x - center_x) ** 2)
    # 将圆弧上的点转换为整数坐标，并加上垂直距离
    arc_points_x = np.round(arc_points_x).astype(int)
    arc_points_y = np.round(arc_points_y).astype(int)
    # 组合成坐标对
    arc_points = np.column_stack((arc_points_x, arc_points_y))
    return arc_points, center_point, radius


def calculate_tangent_line(center_point, arc_points):
    slope = []
    # 圆心坐标
    center_x = center_point[0]
    center_z = center_point[2]
    for point in arc_points:
        # 点的坐标
        x1, z1 = point
        # 计算过圆心的斜率
        slope = (z1-center_z) / (x1-center_x)
        slope.append(slope)
    return slope

def get_y_points(cbct_img):
    cbct_heigh, cbct_width, cbct_depth = np.shape(cbct_img)  # (xyz)
    xz_points = []
    for x in range(cbct_width):
        for z in range(cbct_depth):
            xz_points.append((x, z))
    return xz_points


def find_intersection_points(k, b, w, h):
    intersections = []
    # 计算矩形边界与射线的交点，并将其添加到列表中
    # x = 0
    x_intersection = 0
    y_intersection = k * x_intersection + b
    if 0 <= y_intersection <= h:
        if (x_intersection, y_intersection) not in intersections:
            intersections.append((x_intersection, y_intersection))
    # x = w
    x_intersection = w
    y_intersection = k * x_intersection + b
    if 0 <= y_intersection <= h:
        if (x_intersection, y_intersection) not in intersections:
            intersections.append((x_intersection, y_intersection))
    # y = 0
    y_intersection = 0
    x_intersection = (y_intersection - b)/k
    if 0 <= x_intersection <= w:
        if (x_intersection, y_intersection) not in intersections:
            intersections.append((x_intersection, y_intersection))
    # y = h
    y_intersection = h
    x_intersection = (y_intersection - b)/k
    if 0 <= x_intersection <= w:
        if (x_intersection, y_intersection) not in intersections:
            intersections.append((x_intersection, y_intersection))

    if len(intersections) == 2:
        pass
    else:
        print('只有一个交点！')

    return intersections


def get_line_points(inter_points, cbct_heigh, cbct_depth):
    # 如果有两个交点，对交点之间的线段进行均匀采样
    line_points = set()

    x1, y1 = inter_points[0]
    x2, y2 = inter_points[1]
    # num_samples = 800  # 采样点数量
    # 计算线段的长度
    line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    # 根据线段的长度确定采样点的数量
    num_samples = int(line_length) + 1
    for i in range(1, num_samples):
        x = round(x1 + (x2 - x1) * i / num_samples)
        y = round(y1 + (y2 - y1) * i / num_samples)
        if (x, y) not in line_points:
            # 判断采样点是否超出边界范围
            if 0 <= x < cbct_heigh and 0 <= y < cbct_heigh:
                line_points.add((x, y))
            else:
                print(f'采样点 ({x}, {y}) 超出边界范围！')
    return list(line_points)


def get_PX_from_cbct(cbct_image, arc_points, center_point, axial_mip):
    cbct_heigh, cbct_width, cbct_depth = np.shape(cbct_image)  # 768,768,576
    # 圆心坐标
    center_x, center_y, center_z = center_point   # 385,385,288
    px_sit_all = []
    mapping_table = {}
    project_image = np.zeros((len(arc_points), 512))  # 512,576
    for k, arc_point in enumerate(arc_points):
        # 计算射线穿过 CBCT 图像的 x 和 z 坐标
        x1, z1 = arc_point
        # 计算过圆心的斜率
        slope = (z1 - center_y) / (x1 - center_x)
        b = z1 - slope * x1
        '''get 射线与矩形交点'''
        inter_points = find_intersection_points(slope, b, cbct_width, cbct_heigh)
        '''两交点密集取点'''
        line_points = get_line_points(inter_points, cbct_heigh, cbct_depth)
        if MODE == 'test':
            line_points_show = np.array(line_points)
            show_center_curve_bounds(axial_mip, arc_points, line_points_show, center_point, title='all points')
        '''每一个depth 取平面上的线段点的平均'''
        # for cbct_z in range(cbct_depth):
        for cbct_z in range(512):
            px_values = []
            if cbct_depth < cbct_z:
                project_image[k][cbct_z] = 0
            else:
                for i, point in enumerate(line_points):
                    cbct_x, cbct_y = point
                    px_sit_all.append((cbct_y, cbct_x, cbct_z))
                    px_values.append(cbct_image[cbct_y, cbct_x, cbct_z])
                    mapping_table[(cbct_y, cbct_x, cbct_z)] = [(k, cbct_z)]
                # project_image[k][cbct_z] = np.mean(px_values)
                project_image[k][cbct_z] = np.max(px_values)
        print('finished one arc point = ', k)
    return project_image, mapping_table



def save_mapping_table(mapping_table, filename, casename):
    path = join_path(filename, casename)
    with open(path, 'wb') as f:
        pickle.dump(mapping_table, f)


'''load pkl'''
def load_mapping_table(filename):
    with open(filename, 'rb') as f:
        mapping_table = pickle.load(f)
    return mapping_table



# generate panoramic x-ray image important!
def prepare_mat(data_id, nii_path, MODE):
    # load nii
    case_name = data_id
    # case_path = 'xxx/DATA/nii_gz/nii_raw/CASE_ID.nii.gz'
    case_path = join_path(nii_path, data_id + '.nii.gz')
    print("\n case_name = " + case_name)
    cbct_image, affine_matrix = read_nii(case_path, AFFINE=True)

    # Step 1: get 圆弧坐标
    # cbct_copy = cbct_image.copy()
    '''get 圆弧上1000个点集/圆心/半径'''
    arc_points, center_point, radius = generate_arc_points(cbct_image, num_points=512)
    axial_mip = generate_MIP(cbct_image, direction='axial')
    if MODE == 'test':
        show_center_bounds(axial_mip, arc_points, title='arc_points Curve')
    coronal_mip = generate_MIP(cbct_image, direction='coronal')
    if MODE == 'test':
        show_rotate(coronal_mip, 'Coronal MIP')
    #
    # '''get 斜率'''
    # slope = calculate_tangent_line(center_point, arc_points)
    # '''get 平面点集'''
    # XZ_points = get_y_points(cbct_image)
    '''get PX'''
    PX, mapping_table = get_PX_from_cbct(cbct_image, arc_points, center_point, axial_mip)

    if MODE == 'run':
        show_mask(PX, 'Axial MIP')
    save_png(PX, ideal_px_dir, case_name + '.png')
    print('shape = ', PX.shape)
    px_img_ideal = norm_px(PX)
    px_img_ideal = px2img(px_img_ideal)
    print('save img !')
    save_png(px_img_ideal, final_px_dir, case_name + '.png')
    '''save mapping table'''
    print('save mapping table')
    save_mapping_table(mapping_table, pkl_dir, case_name + '.pkl')


    # # Step 7: generate bone and MPR images
    # cbct_copy = cbct_image.copy()
    # bone_mask = cbct_copy > 1500
    # # save_nii(bone_mask.astype(np.uint8), affine_matrix, case_name+'.nii.gz', tmp_bone_dir)
    #
    # # 下面这句mask 很多信息 全景片比较黑 舍去
    # # cbct_image[bone_mask == 0] = 0
    # MPR_images = get_MPR_images(cbct_image, sample_bounds, arch_thickness)
    # show_result(MPR_images[0], 'MPR Image')
    # show_result(MPR_images[20], 'MPR Image')
    # show_result(MPR_images[40], 'MPR Image')
    # px_img_ideal = assemble_MPR_images(MPR_images)
    # px_img_ideal = norm_px(px_img_ideal)
    # px_img_ideal = px2img(px_img_ideal)
    #
    # show_result(px_img_ideal, 'Ideal Panoramic Image')
    # img_path = join_path(ideal_px_dir, case_name + '.png')
    # px_img_ideal = cv2.imread(img_path)
    # px_img_ideal = cv2.flip(px_img_ideal, 0)
    # save_png(px_img_ideal, ideal_px_dir, case_name + '.png')
    # # px_img_ideal = cv2.resize(px_img_ideal, (512, 512))
    # # save_png(px_img_ideal, final_px_dir, case_name + '.png')
    # print(px_img_ideal.shape)
    #
    # if MODE == 'test':
    #     show_result(px_img_ideal, 'Ideal Panoramic Image')

    #


# if mode is test, intermediate result will be shown
if __name__ == '__main__':
    nii_path = 'xxx/Project/CBCT_Main/data/knee_cbct/512nii'  # 更改路径
    files = os.listdir(nii_path)
    total_n = len(files)
    MODE = 'run'
    # MODE = 'test'

    '''test'''
    # Call the function to load mapping_table
    # mapping_table = load_mapping_table('xxx/DATA/Panoramic_Mean/mapping_table/CASE_ID.pkl')

    for file in files:
        data_id = file.split('.', 3)[0]
        jpg = str('xxx/DATA_SET/Panoramic_Max/final_px512/' + data_id + ".png")
        # jpg = str('xxx/DATA/Panoramic_Mean/final_px512/CASE_ID.png')
        # image = cv2.imread('xxx/DATA/Panoramic_Mean/final_px512/CASE_ID_test.png')
        # unique_values = np.unique(image)
        # 判断图片是否存在
        if os.path.exists(jpg):
            continue
        if data_id == 'CASE_ID':
            continue
        if data_id == 'CASE_ID':  # 没牙齿
            continue
        else:
            # data_id = 'CASE_ID'
            prepare_mat(data_id, nii_path, MODE)
            # if data_id >= 133:
            #    break
            print("break up ! PatientID = ", data_id)