import trimesh
import numpy as np

# 读取源模型文件
source_mesh = trimesh.load_mesh("xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/Visualize/GTCASE_ID/41._Root.stl")  # 替换成源模型文件路径

# 读取目标模型文件
target_mesh = trimesh.load_mesh("xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/Visualize/CASE_ID/teeth_205.ply")  # 替换成目标模型文件路径

# 获取源模型的坐标和角度信息
source_translation = source_mesh.vertices.mean(axis=0)  # 获取坐标信息
source_rotation_matrix = source_mesh.vertices.mean(axis=0)  # 获取角度信息

# 将源模型的坐标和角度信息应用到目标模型上
target_mesh.apply_translation(-source_translation)  # 平移
target_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, source_rotation_matrix))  # 旋转

# 保存结果
target_mesh.export("xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/Visualize/CASE_ID/transformed_target_model.obj")