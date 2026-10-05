# reconstruct_tooth_mesh.py
# 把牙齿的 obj 文件点云重建成 watertight mesh 并保存

import copy
import numpy as np
import open3d as o3d
import trimesh
import os


def showPointCloud(vertices, windowName=""):
    """Visualize the pointcloud given the 3D points
    vertices: numpy array, shape (N,3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=windowName,
        width=800,
        height=600,
        left=50,
        top=50,
        point_show_normal=False,
    )


def farthestPointDownSample(vertices, num_point_sampled, return_flag=False):
    """Farthest Point Sampling (FPS) algorithm
    Input:
        vertices: numpy array, shape (N,3) or (N,2)
        num_point_sampled: int, the number of points after downsampling, should be no greater than N
        return_flag: bool, whether to return the mask of the selected points
    Output:
        selected_vertices: numpy array, shape (num_point_sampled,3) or (num_point_sampled,2)
        [Optional] flags: boolean numpy array, shape (N,)
    """
    N = len(vertices)
    n = num_point_sampled
    assert n <= N, "Num of sampled point should be <= size of vertices."

    # 先找距离整体质心最远的点作为起点
    _G = np.mean(vertices, axis=0)  # centroid of vertices
    _d = np.linalg.norm(vertices - _G, axis=1, ord=2)
    farthest = np.argmax(_d)

    distances = np.inf * np.ones((N,))
    flags = np.zeros((N,), np.bool_)

    for _ in range(n):
        flags[farthest] = True
        distances[farthest] = 0.0
        p_farthest = vertices[farthest]
        dists = np.linalg.norm(vertices[~flags] - p_farthest, axis=1, ord=2)
        distances[~flags] = np.minimum(distances[~flags], dists)
        farthest = np.argmax(distances)

    if return_flag:
        return vertices[flags], flags
    else:
        return vertices[flags]


def surfaceVertices2WatertightO3dMesh(vertices, showInWindow=False):
    """Construct single tooth triangle mesh from surface vertices by Poisson surface reconstruction
    Input:
        vertices: numpy array, shape (N,3)
        showInWindow: bool, whether to visualize the constructed mesh
    Output:
        mesh: open3d.geometry.TriangleMesh, the constructed mesh of a single tooth
    """
    # 构建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # 1. 估计法向量（邻域内点的 PCA）
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )

    # 2. 使用切平面一致性方法，使局部法线方向一致
    pcd.orient_normals_consistent_tangent_plane(k=30)

    # 3. 按论文思路，用整体质心来统一“朝外 / 朝里”方向
    pcd.normalize_normals()
    normal = np.asarray(pcd.normals)
    center = np.mean(vertices, axis=0)
    vecs = vertices - center        # 质心指向各点的向量
    # 如果法线和 (center→point) 的夹角>90°，就翻转
    opposite_mask = np.sum(vecs * normal, axis=1) < 0
    normal[opposite_mask] = -normal[opposite_mask]
    pcd.normals = o3d.utility.Vector3dVector(normal)

    # 4. Poisson 表面重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=12, scale=1.0
    )

    # 5. 用密度把外面的“浮壳”去掉（常见操作）
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.0005)  # 去掉最稀疏的 1%
    vertices_to_keep = densities > density_threshold
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])

    # 6. 清理一下 mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    if showInWindow:
        mesh.paint_uniform_color(np.array([0.7, 0.7, 0.7]))
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="Open3D reconstructed watertight mesh",
            width=800,
            height=600,
            left=50,
            top=50,
            point_show_normal=False,
            mesh_show_wireframe=True,
            mesh_show_back_face=True,
        )

    return mesh, normal


def exportTriMeshObj(vertices, faces, objFile):
    """Save a triangle mesh in OBJ format
    Input:
        vertices: numpy array, shape (N,3)
        faces: numpy array, shape (M,3)
        objFile: str, file path to save with ".obj" suffix
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    export_str = trimesh.exchange.obj.export_obj(
        mesh,
        include_normals=True,
        include_color=False,
        include_texture=False,
        return_texture=False,
        write_texture=False,
        resolver=None,
        digits=8,
    )
    with open(objFile, "w") as f:
        f.write(export_str)


def savePointCloudWithNormals(vertices, normals, file_path):
    """Save point cloud with normals to a file (e.g., .ply or .pcd)
    vertices: numpy array, shape (N,3)
    normals: numpy array, shape (N,3)
    file_path: str, path to save the point cloud (must end with .ply or .pcd)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Save the point cloud with normals
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"Point cloud with normals saved to {file_path}")


def process_all_obj_files(input_dir, output_dir):
    """Process all .obj files in the subfolders of the given directory"""
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".obj"):
                # 获取当前文件夹下的 .obj 文件
                in_obj_path = os.path.join(root, filename)
                id_name = os.path.basename(root)
                # 为每个 .obj 文件创建一个以其命名的文件夹
                output_subfolder = os.path.join(output_dir, id_name)
                os.makedirs(output_subfolder, exist_ok=True)

                # 设置输出路径
                out_ply_path = os.path.join(output_subfolder, f"{filename.replace('.obj', '_poisson.ply')}")
                out_normal_ply_path = os.path.join(output_subfolder, f"{filename.replace('.obj', '_with_normals.ply')}")

                print(f"Processing {in_obj_path}...")

                # 读取 obj 文件
                tm = trimesh.load(in_obj_path, process=False)
                vertices = tm.vertices.astype(np.float64)
                print(f"Loaded vertices: {vertices.shape}")

                # Poisson 重建成 watertight mesh
                mesh, normals = surfaceVertices2WatertightO3dMesh(vertices, showInWindow=False)

                # 保存为 .ply（Open3D）
                o3d.io.write_triangle_mesh(out_ply_path, mesh)
                print(f"Saved poisson mesh (ply) to: {out_ply_path}")

                # 保存带法线的点云
                savePointCloudWithNormals(vertices, normals, out_normal_ply_path)

                print(f"Finished processing {in_obj_path}")


if __name__ == "__main__":
    # 输入文件夹路径
    input_dir = "xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/visual/0117_CrossAttention_95_Infer/"

    # 输出文件夹路径
    output_dir = "xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/visualize-new/"

    # 处理文件夹中的所有 .obj 文件
    process_all_obj_files(input_dir, output_dir)
