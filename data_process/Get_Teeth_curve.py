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
    bounding_box = voxel_grid.bounding_box
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

    # save 三维数组
    with open("xxx/Project/max_cor.txt", 'w') as outfile:
        for slice_2d in teeth_array:
            np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')
    # # 一维 / 二维数组保存
    # np.savetxt("xxx/Project/max_cor.txt",
    #            teeth_array, fmt='%f', delimiter=',')

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

    # Visualize the projected mesh
    # Target_Up_imge.export('xxx/Project/project.obj')
    # Target_Up_Mesh.show()



    # mesh = trimesh.exchange.export.export_dict(Target_Up_Mesh, encoding=None)
    # # normals =
    # # Define the plane onto which to project the mesh
    # plane_normal = [0, 0, 1]  # z-axis
    # plane_origin = [0, 0, 0]  # origin of the plane
    #
    # # plane = trimesh.scene.Plane(plane_normal, plane_origin)
    # # plane = trimesh.geometry.Plane(plane_normal, plane_origin)
    #
    # # Project the mesh onto the plane
    # # normal = Target_Up_Mesh








    # Compute convex hull
    convex_hull = point_cloud.convex_hull

    # Project convex hull onto x-y plane
    projected_hull = convex_hull.project_to_2D()

    # Extract x-y coordinates
    xy_coords = projected_hull.vertices[:, :2]

    # data3d = np.asarray(Target_Transform_Result_Up.points)
    # # 点云读入 o3d.t.io
    # Target_Transform_Result_Up = o3d.t.io.read_point_cloud('xxx/Project/Target_Transform_Result_Up.ply')
    # print(Target_Transform_Result_Up)
    # # tensor64 = o3d.core.Tensor(data3d)
    # # # create a float32 Tensor and copy data3d from the float64 Tensor
    # # tensor32 = o3d.core.Tensor(np.empty_like(data3d, dtype=np.float32))
    # # data32 = tensor64.GetData().astype(np.float32)
    # # tensor32.SetData(tensor64.GetDataPtrAsFloat64(), copy=True)
    #
    # intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6],
    #                              [0, 0, 1]])
    # rgbd_reproj = Target_Transform_Result_Up.project_to_rgbd_image(640,
    #                                         480,
    #                                         intrinsic,
    #                                         depth_scale=5000.0,
    #                                         depth_max=10.0)
    #
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(np.asarray(rgbd_reproj.color.to_legacy()))
    # axs[1].imshow(np.asarray(rgbd_reproj.depth.to_legacy()))
    # plt.show()










    # Generate a random 3D point cloud
    points_arr = np.asarray(Target_Transform_Result_Up.points)
    print('arr = ', points_arr)
    # 垂直投影

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_arr)

    # Define the plane equation for the xy plane
    plane_eq = np.array([0, 0, 1, 0])  # [a, b, c, d] where ax + by + cz + d = 0

    # Project the point cloud onto the xy plane
    pcd_projected, _ = pcd.project_to_rgbd_image(plane_eq)

    # Visualize the results
    o3d.visualization.draw_geometries([pcd, pcd_projected])

    # Create a plane that the points will be projected onto
    plane_normal = [0, 0, 1]
    plane_origin = [0, 0, 0]
    plane = trimesh.create.box(plane_normal, plane_origin)

    # Project the points onto the plane
    # points_2d = plane.project_points(points)

    # o3d.io.write_point_cloud('xxx/Project/curve.ply', points_2d)


    # # 1.0scale
    # source_temp1 = o3d.io.read_triangle_mesh(single_source_path1)
    # source_temp1 = copy.deepcopy(source_temp1)
    # source_temp1 = source_temp1.sample_points_uniformly(number_of_points=200000)
    # source_temp1 = source_temp1.paint_uniform_color([1, 0, 1])  # pink
    # retest = source_temp + source_temp1
    # o3d.io.write_point_cloud('xxx/Project/before_scale.ply', retest)
    # scale_factor = 1/0.3  # 前4位
    # print(scale_factor)
    # source_temp.scale(scale_factor, center=(0, 0, 0))
    # result = source_temp + source_temp1
    # o3d.io.write_point_cloud('xxx/Project/scale_test.ply', result)
