#############
# Seg_nrrd TO mesh
#############

from utils.local_io import *
import numpy as np
import trimesh
from skimage import measure
import nibabel as nib

def vol_to_mesh(vol, spacing, step_size=1):
    a, b, c = np.where(vol > 0)
    if len(a) == 0 or len(b) == 0 or len(c) == 0:
        return trimesh.Trimesh()
    a1, a2 = a.min(), a.max() + 1
    b1, b2 = b.min(), b.max() + 1
    c1, c2 = c.min(), c.max() + 1
    vol = vol[a1:a2, b1:b2, c1:c2]

    pad_width = 5
    vol = np.pad(vol, pad_width=pad_width)
    verts, faces, normals, values = measure.marching_cubes(
        vol,
        gradient_direction='ascent',
        spacing=spacing,
        step_size=step_size
    )
    verts += (np.array([a1, b1, c1]) - pad_width) * spacing
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh


##################################
#  读取一下原始cbct的nrrd 获得里面的space信息
##################################
def read_nrrd(data_id):
    # load nrrd
    print('patient id = ', data_id)
    root_path = 'xxx/DATA/Seg_Teeth_nrrd/teeth'  # 分割之后的nrrd
    case_path = os.path.join(root_path, data_id + '_teeth_1022.seg.nrrd')
    # root_path = 'xxx/DATA/nrrd_data_500'   # 原始的nrrd
    # case_path = os.path.join(root_path, data_id + '.nrrd')
    print("\n case_path = " + case_path)
    options = nrrd.read(case_path)
    spacing = options[1]['space directions']
    x = spacing[0][0]
    y = spacing[1][1]
    z = spacing[2][2]
    translation = np.array([x, y, z])
    print(translation)
    return translation

def nrrd2mesh(data_id, myparth, translation):
    # load nrrd
    case_name = data_id
    print(translation)
    # case_path = 'xxx/DATA/Seg_nrrd/teeth/CASE_ID_teeth_1022.seg.nrrd'
    case_path = join_path(myparth, data_id + '_teeth_1022.seg.nrrd')
    print("\n case_name = " + case_name)
    data, options = nrrd.read(case_path)
    data_root = 'xxx/DATA/Seg_Teeth_Mesh_Scale'
    mesh_dir = check_dir(join_path(data_root, data_id))
    teeth_unique = np.unique(data)
    mask = teeth_unique != 0  # [False True]
    teeth_unique = teeth_unique[mask]
    print(teeth_unique)
    total_n = len(teeth_unique)
    print(total_n)
    # get 3d arr shape
    shape0 = data.shape[0]
    shape1 = data.shape[1]
    shape2 = data.shape[2]
    print('shape0=', shape0, 'shape1=',  shape1, 'shape2=', shape2)
    # 建立 0 三维矩阵
    teeth_map1 = np.zeros((shape0, shape1, shape2))
    teeth_map2 = np.zeros((shape0, shape1, shape2))
    teeth_map3 = np.zeros((shape0, shape1, shape2))
    teeth_map4 = np.zeros((shape0, shape1, shape2))
    teeth_map5 = np.zeros((shape0, shape1, shape2))
    teeth_map6 = np.zeros((shape0, shape1, shape2))
    teeth_map7 = np.zeros((shape0, shape1, shape2))
    teeth_map8 = np.zeros((shape0, shape1, shape2))
    teeth_map9 = np.zeros((shape0, shape1, shape2))
    teeth_map10 = np.zeros((shape0, shape1, shape2))
    teeth_map11 = np.zeros((shape0, shape1, shape2))
    teeth_map12 = np.zeros((shape0, shape1, shape2))
    teeth_map13 = np.zeros((shape0, shape1, shape2))
    teeth_map14 = np.zeros((shape0, shape1, shape2))
    teeth_map15 = np.zeros((shape0, shape1, shape2))
    teeth_map16 = np.zeros((shape0, shape1, shape2))
    teeth_map17 = np.zeros((shape0, shape1, shape2))
    teeth_map18 = np.zeros((shape0, shape1, shape2))
    teeth_map19 = np.zeros((shape0, shape1, shape2))
    teeth_map20 = np.zeros((shape0, shape1, shape2))
    teeth_map21 = np.zeros((shape0, shape1, shape2))
    teeth_map22 = np.zeros((shape0, shape1, shape2))
    teeth_map23 = np.zeros((shape0, shape1, shape2))
    teeth_map24 = np.zeros((shape0, shape1, shape2))
    teeth_map25 = np.zeros((shape0, shape1, shape2))
    teeth_map26 = np.zeros((shape0, shape1, shape2))
    teeth_map27 = np.zeros((shape0, shape1, shape2))
    teeth_map28 = np.zeros((shape0, shape1, shape2))
    teeth_map29 = np.zeros((shape0, shape1, shape2))
    teeth_map30 = np.zeros((shape0, shape1, shape2))
    teeth_map31 = np.zeros((shape0, shape1, shape2))
    teeth_map32 = np.zeros((shape0, shape1, shape2))
    teeth_map33 = np.zeros((shape0, shape1, shape2))
    teeth_map34 = np.zeros((shape0, shape1, shape2))
    teeth_map35 = np.zeros((shape0, shape1, shape2))
    teeth_map36 = np.zeros((shape0, shape1, shape2))
    teeth_map37 = np.zeros((shape0, shape1, shape2))
    teeth_map38 = np.zeros((shape0, shape1, shape2))
    teeth_map39 = np.zeros((shape0, shape1, shape2))
    teeth_map40 = np.zeros((shape0, shape1, shape2))

    for index, x in np.ndenumerate(data):
        if x != 0:
            if x == 1:
                # print(index, x)
                teeth_map1[index] = x
                # print(teeth_map1)
                #print(index, 'teeth_map1 = ', teeth_map1[index])
                #print(teeth_map1.shape)
            elif x == 2:
                # print(index, teeth_map[index])
                teeth_map2[index] = x
                #print(index, 'teeth_map2 = ', teeth_map2[index])
            elif x == 3:
                # print(index, teeth_map[index])
                teeth_map3[index] = x
                #print(index, 'teeth_map3 = ', teeth_map3[index])
            elif x == 4:
                # print(index, teeth_map[index])
                teeth_map4[index] = x
                #print(index, 'teeth_map4 = ', teeth_map4[index])
            elif x == 5:
                # print(index, teeth_map[index])
                teeth_map5[index] = x
                #print(index, 'teeth_map5 = ', teeth_map5[index])
            elif x == 6:
                # print(index, teeth_map[index])
                teeth_map6[index] = x
                #print(index, 'teeth_map6 = ', teeth_map6[index])
            elif x == 7:
                # print(index, teeth_map[index])
                teeth_map7[index] = x
                #print(index, 'teeth_map7 = ', teeth_map7[index])
            elif x == 8:
                # print(index, teeth_map[index])
                teeth_map8[index] = x
                #print(index, 'teeth_map8 = ', teeth_map8[index])
            elif x == 9:
                # print(index, teeth_map[index])
                teeth_map9[index] = x
                #print(index, 'teeth_map9 = ', teeth_map9[index])
            elif x == 10:
                # print(index, teeth_map[index])
                teeth_map10[index] = x
                #print(index, 'teeth_map10 = ', teeth_map10[index])
            elif x == 11:
                # print(index, teeth_map[index])
                teeth_map11[index] = x
                #print(index, 'teeth_map11 = ', teeth_map11[index])
            elif x == 12:
                # print(index, teeth_map[index])
                teeth_map12[index] = x
                #print(index, 'teeth_map12 = ', teeth_map12[index])
            elif x == 13:
                # print(index, teeth_map[index])
                teeth_map13[index] = x
                #print(index, 'teeth_map13 = ', teeth_map13[index])
            elif x == 14:
                # print(index, teeth_map[index])
                teeth_map14[index] = x
                #print(index, 'teeth_map14 = ', teeth_map14[index])
            elif x == 15:
                # print(index, teeth_map[index])
                teeth_map15[index] = x
                #print(index, 'teeth_map15 = ', teeth_map15[index])
            elif x == 16:
                # print(index, teeth_map[index])
                teeth_map16[index] = x
                #print(index, 'teeth_map16 = ', teeth_map16[index])
            elif x == 17:
                # print(index, teeth_map[index])
                teeth_map17[index] = x
                #print(index, 'teeth_map17 = ', teeth_map17[index])
            elif x == 18:
                # print(index, teeth_map[index])
                teeth_map18[index] = x
                #print(index, 'teeth_map18 = ', teeth_map18[index])
            elif x == 19:
                # print(index, teeth_map[index])
                teeth_map19[index] = x
                #print(index, 'teeth_map19 = ', teeth_map19[index])
            elif x == 20:
                # print(index, teeth_map[index])
                teeth_map20[index] = x
                #print(index, 'teeth_map20 = ', teeth_map20[index])
            elif x == 21:
                # print(index, teeth_map[index])
                teeth_map21[index] = x
                #print(index, 'teeth_map21 = ', teeth_map21[index])
            elif x == 22:
                # print(index, teeth_map[index])
                teeth_map22[index] = x
                #print(index, 'teeth_map22 = ', teeth_map22[index])
            elif x == 23:
                # print(index, teeth_map[index])
                teeth_map23[index] = x
                #print(index, 'teeth_map23 = ', teeth_map23[index])
            elif x == 24:
                # print(index, teeth_map[index])
                teeth_map24[index] = x
                #print(index, 'teeth_map24 = ', teeth_map24[index])
            elif x == 25:
                # print(index, teeth_map[index])
                teeth_map25[index] = x
                #print(index, 'teeth_map25 = ', teeth_map25[index])
            elif x == 26:
                # print(index, teeth_map[index])
                teeth_map26[index] = x
                #print(index, 'teeth_map26 = ', teeth_map26[index])
            elif x == 27:
                # print(index, teeth_map[index])
                teeth_map27[index] = x
                #print(index, 'teeth_map27 = ', teeth_map27[index])
            elif x == 28:
                # print(index, teeth_map[index])
                teeth_map28[index] = x
                #print(index, 'teeth_map28 = ', teeth_map28[index])
            elif x == 29:
                # print(index, teeth_map[index])
                teeth_map29[index] = x
                #print(index, 'teeth_map29 = ', teeth_map29[index])
            elif x == 30:
                # print(index, teeth_map[index])
                teeth_map30[index] = x
                #print(index, 'teeth_map30 = ', teeth_map30[index])
            elif x == 31:
                # print(index, teeth_map[index])
                teeth_map31[index] = x
                #print(index, 'teeth_map31 = ', teeth_map31[index])
            elif x == 32:
                # print(index, teeth_map[index])
                teeth_map32[index] = x
                #print(index, 'teeth_map32 = ', teeth_map32[index])
            elif x == 33:
                # print(index, teeth_map[index])
                teeth_map33[index] = x
                #print(index, 'teeth_map33 = ', teeth_map33[index])
            elif x == 34:
                # print(index, teeth_map[index])
                teeth_map34[index] = x
                #print(index, 'teeth_map34 = ', teeth_map34[index])
            elif x == 35:
                # print(index, teeth_map[index])
                teeth_map35[index] = x
                #print(index, 'teeth_map35 = ', teeth_map35[index])
            elif x == 36:
                # print(index, teeth_map[index])
                teeth_map36[index] = x
                #print(index, 'teeth_map36 = ', teeth_map36[index])
            elif x == 37:
                # print(index, teeth_map[index])
                teeth_map37[index] = x
                #print(index, 'teeth_map37 = ', teeth_map37[index])
            elif x == 38:
                # print(index, teeth_map[index])
                teeth_map38[index] = x
                #print(index, 'teeth_map38 = ', teeth_map38[index])
            elif x == 39:
                # print(index, teeth_map[index])
                teeth_map39[index] = x
                #print(index, 'teeth_map39 = ', teeth_map39[index])
            elif x == 40:
                # print(index, teeth_map[index])
                teeth_map40[index] = x
                #print(index, 'teeth_map40 = ', teeth_map40[index])

            else:
                print('############################# x = ', x)

    for i in range(40):
        i = i + 1
        teeth_map_str = "teeth_map" + str(i)
        teeth_map = eval(teeth_map_str)
        # print(teeth_map.shape)
        teeth_unique = np.unique(teeth_map)
        print(i, 'teeth_unique = ', teeth_unique)
        if all(item == 0 for item in teeth_unique):
            # 👇️ this runs
            print('NULL VOL')
        else:
            print('translation = ', translation)
            mesh = vol_to_mesh(teeth_map, spacing=translation, step_size=1)
            mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5,
                                                      iterations=10,
                                                      implicit_time_integration=False,
                                                      volume_constraint=True,
                                                      laplacian_operator=None)
            mesh_single_dir = join_path(mesh_dir, str(i) + '_Root.stl')
            mesh.export(mesh_single_dir)
            print('save finished -- ' + teeth_map_str)





#######备用 单个牙齿循环
        # for teeth_id in teeth_unique:
        #     teeth_map = data.copy()
        #     # teeth_single =
        #     for index, x in np.ndenumerate(data):
        #         if x != 0:
        #             if x == teeth_id:
        #                 # print(index, x)
        #                 teeth_map[index] = x
        #                 print(teeth_map[index])
        #             else:
        #                 # print(index, teeth_map[index])
        #                 teeth_map[index] = 0
        #                 print(index, teeth_map[index])
        #     # print(teeth_map)
        #     teeth_unique = np.unique(teeth_map)
        #     print(teeth_unique)
        #     mesh = vol_to_mesh(teeth_map, spacing=(0.1, 0.1, 0.1), step_size=1)
        #     # mesh = mesh.laplacian_smooth(cotangentweight=False)
        #     mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=10, implicit_time_integration=False,
        #                                               volume_constraint=True, laplacian_operator=None)
        #     data_id = 'C1000233434'
        #     data_root = 'xxx/DATA/Teeth_Seg_Mesh'
        #     mesh_dir = check_dir(join_path(data_root, data_id))
        #     mesh_single_dir = join_path(mesh_dir, str(teeth_id) + '_Root.stl')
        #     mesh.export(mesh_single_dir)
#######备用 单个牙齿循环

if __name__ == '__main__':

    myparth = 'xxx/DATA/Seg_Teeth_nrrd/teeth'  # 更改路径
    # myparth = 'xxx/DATA/nii_gz/nii_raw/'
    files = os.listdir(myparth)
    total_n = len(files)
    for file in files:
        data_id = file.split('_', 3)[0]
        data_id = 'CASE_ID'  # test
        patient_path = str('xxx/DATA/Seg_Teeth_Mesh_Scale/' + data_id)

        # 判断是否存在
        if os.path.exists(patient_path):
            continue
        else:
            translation = read_nrrd(data_id)
            nrrd2mesh(data_id, myparth, translation)
            print("Finished One Case !   PatientID = ", data_id)
    print("finished all！")

    # data = np.random.random((2, 3, 4))
    # print(data.shape)