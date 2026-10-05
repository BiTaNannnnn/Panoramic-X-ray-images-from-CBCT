import open3d as o3d
import numpy as np
import trimesh
import copy
import cv2 as cv
from trimesh.voxel import creation
import matplotlib.pyplot as plt





def Target_Transform(target_path, transformation):
    target_up_mesh = o3d.io.read_triangle_mesh(target_path)
    # mesh 2 pcd
    # target_up_mesh = target_up_mesh.sample_points_uniformly(number_of_points=200000)
    target_up_mesh = copy.deepcopy(target_up_mesh)
    print('transformation = ', transformation)
    transformation_ni = np.linalg.inv(transformation)
    print(transformation_ni)
    # 逆矩阵
    target_up_mesh = target_up_mesh.transform(transformation_ni)
    target_up_mesh = target_up_mesh.paint_uniform_color([1, 0, 1])  # pink
    # 1/spacing 转换成原始1.0大小 与 nrrd 数据重合
    scale_factor = 1 / 0.3
    print(scale_factor)
    target_up_mesh.scale(scale_factor, center=(0, 0, 0))
    # o3d.io.write_triangle_mesh('xxx/Project/target_ni.ply', target_up_mesh)
    o3d.io.write_triangle_mesh('xxx/Project/Target_mesh_Up.ply', target_up_mesh)

    return target_up_mesh


if __name__ == '__main__':

    # load pcd
    target_path = 'xxx/DATA/CBCT_data_2021_8_13_Processed/CASE_ID/Up_Root.stl'
    single_source_path = 'xxx/DATA/Seg_Teeth_Mesh_Scale_test/CASE_ID/all_Root.stl'
    single_source_path1 = 'xxx/DATA/Seg_Teeth_Mesh_Scale_test/CASE_ID/all_Root_1.0.stl'

    transformation = np.loadtxt("xxx/DATA/Seg_Teeth_Mesh_Scale_test/CASE_ID/Registration_matrix_Up.txt", delimiter=',')
    Target_Transform_Result_Up = Target_Transform(target_path, transformation)

    # test result 0.1scale
    source_temp = o3d.io.read_triangle_mesh(single_source_path1)
    source_temp = copy.deepcopy(source_temp)
    # mesh to pcd
    # source_temp = source_temp.sample_points_uniformly(number_of_points=200000)
    source_temp = source_temp.paint_uniform_color([1, 1, 0])  # yellow

    # test： target 和原始nrrd数据重合
    result = Target_Transform_Result_Up + source_temp
    o3d.io.write_triangle_mesh('xxx/Project/up_transform_test.ply', result)





    #  load trimesh mesh
    voxel_grid = trimesh.load('xxx/Project/Target_mesh_Up.ply')
    bounding_box_min = voxel_grid.bounding_box.bounds[0]
    bounding_box_max = voxel_grid.bounding_box.bounds[1]
    # bounding_box_min.export('xxx/Project/bounding_box.ply')
    # trimesh体素化
    voxel_grid = creation.voxelize(voxel_grid, 1)
    # print(transformation)
    # voxel_grid = voxel_grid.transform(transformation)


    # # 体素平移到box里
    # voxel_grid = voxel_grid.marching_cubes
    # voxel_grid.export('xxx/Project/voxel_grid.ply')
    # 密集矩阵
    voxel_grid = voxel_grid.matrix
    voxel_grid_unique = np.unique(voxel_grid)
    print(voxel_grid_unique)
    # voxel_grid = voxel_grid.marching_cubes
    # voxel_grid.export('xxx/Project/voxel_grid.ply')

    teeth_array = np.zeros((768, 768, 576), dtype='int')
    teeth_array[296:515, 145:302, 232:342] = voxel_grid
    # teeth_array[296][145][232] = voxel_grid
    voxel_grid_unique = np.unique(voxel_grid)
    print(teeth_array)
    print(voxel_grid_unique)


    # arry在xy从上到下的投影 max result
    max_time = -100000
    max_cor = np.empty_like(copy.deepcopy(teeth_array[:, :, 0]))
    for i in range(0, max_cor.shape[0]):
        for j in range(0, max_cor.shape[1]):
            max_cor[i][j] = max(teeth_array[i, j, :])
    print("finish")

    max_cor = np.rot90(max_cor, 1)
    max_cor = np.rot90(max_cor, 1)
    max_cor = np.rot90(max_cor, 1)

    plt.imshow(max_cor, "gray")
    plt.axis('off')
    path = 'xxx/Project/001.jpg'
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

    print(voxel_grid)







    # # 换成体素
    #
    # Target_Transform_Result_Up = Target_Transform_Result_Up.compute_vertex_normals()
    #
    # # Fit to unit cube.
    # Target_Transform_Result_Up.scale(1 / np.max(Target_Transform_Result_Up.get_max_bound() - Target_Transform_Result_Up.get_min_bound()),
    #            center=Target_Transform_Result_Up.get_center())
    # # Target_Transform_Result_Up.scale(1 / np.max(Target_Transform_Result_Up.get_max_bound() - Target_Transform_Result_Up.get_min_bound()),
    # #            center=Target_Transform_Result_Up.get_center())
    # print('Displaying input mesh ...')
    # # o3d.visualization.draw([Target_Transform_Result_Up])
    #
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(Target_Transform_Result_Up, voxel_size=0.05)
    # o3d.io.write_voxel_grid('xxx/Project/voxel_grid1.ply', voxel_grid)







    #trimesh 读入 从上到下投影
    # Load point cloud data3d
    Target_Up_Mesh = trimesh.load('xxx/Project/Target_mesh_Up.ply')
    normal = (0, 0, 1)
    Target_Up_imge = Target_Up_Mesh.projected(normal, max_regions=200000)
    Target_Up_imge.show()
    Target_Up_imge = Target_Up_imge.plot_discrete
    Target_Up_imge.show()
    # cv.imwrite('xxx/Project/mask.png', Target_Up_imge)


    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    pts = Target_Up_imge.polygons_closed
    print(pts)
    # pts = np.array([[20, 10], [10, 27], [20, 44], [40, 44], [50, 27], [40, 10]], np.int32)
    img = cv.polylines(canvas, [pts], True, (0, 255, 255))
    # img = np.array(img)
    # img[img != 0] = 1  # 图像二值化
    # plt.subplot(1, 3, 1)
    # plt.imshow(np.rot90(img, k=-1))
    # cv.fillPoly(canvas, [pts], True, (255, 255, 255))
    cv.imwrite('xxx/Project/mask.png', canvas)

    cv.imshow('polyline', canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print('finished')










