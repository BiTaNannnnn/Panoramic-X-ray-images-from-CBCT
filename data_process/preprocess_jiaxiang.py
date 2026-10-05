import os
import glob
import numpy as np
import pandas as pd
import scipy
import trimesh
import pickle
## Env: conda_python3

from scipy.spatial.transform import Rotation as R



def load_mesh(mesh_file):
    mesh = trimesh.load_mesh(mesh_file)
    return mesh

def quaternion_to_matrix(vector):
    M = np.zeros(shape=(4,4))
    M[:3,3]=vector[4:]
    r = R.from_quat(vector[0:4])
    #Origin M[:3,:3]=r.as_dcm()
    M[:3,:3] = r.as_matrix()
    M[3,3]=1
    return M


# # Load meshTeethAxis_Ori


all_stl_files = glob.glob("./**/*.stl", recursive=True)



df_stl = pd.DataFrame(all_stl_files,columns=['file_name'])
# Axis_data_*/ ** / *.stl will be in the column 'folder','sid','stl_file'

df_stl[['sid','stl_file']] = df_stl.file_name.str.split('\\',expand=True).iloc[:,[-2,-1]]  ## in windows use \ instead / in other os

# Set the teeth_id for each stl file. the code extract the number from the file names 
# Origin: df_stl['teeth_id'] = df_stl.stl_file.str.split('.',expand=True).iloc[:,0].astype('int')
df_stl['teeth_id'] = df_stl.stl_file.str.split('.',expand=True).iloc[:,0]
df_stl = df_stl.sort_values(by=['sid','teeth_id']).reset_index(drop=True)
print(df_stl)


save_path = '../world_data/'
# save_path = '../../../../fyp/hzl/data3d/processed_data2/'
os.makedirs(save_path,exist_ok=True)




for ind, row in df_stl.iterrows():
    

    mesh_file = row.file_name
    axis_file = os.path.dirname(mesh_file) + '/Location.txt'
    # print(axis_file)
    # print(mesh_file)
    if mesh_file[-8:-4]== 'Root': 
        #print('copy')
        tid = int(row.teeth_id)
        mesh_local = load_mesh(mesh_file)
        ## load axis
        
        # Origin: axis_file = os.path.dirname(mesh_file) + '/TeethAxis.txt'
        axis_df = pd.read_csv(axis_file,names=['id','q1','q2','q3','w','t1','t2','t3'],sep=' ')
        # axis_df = pd.read_csv(axis_file,names=['id','t1','t2','t3','q1','q2','q3','w'],sep=' ')
        # print( axis_df[axis_df.id==tid])
        trans_vector = axis_df[axis_df.id==tid].to_numpy()[0][1:]
        #########################
        ## reorder: t1,t2,t3,q1,q2,q3,w -> q1,q2,q3,w,t1,t2,t3
        ## dataset2 only
        #print(trans_vector)
        trans_vector = trans_vector[[3,4,5,6,0,1,2]]
        #print(trans_vector)
        #########################
        ## transform mesh
        rotation_local2world = quaternion_to_matrix(trans_vector)
        mesh_world = mesh_local.copy()
        mesh_world.apply_transform(rotation_local2world)
        #print(rotation_local2world)
        rotation_world2local = np.linalg.inv(rotation_local2world)
        # print(rotation_world2local)
        ##
        if not os.path.exists(save_path + row.sid):
            os.makedirs(save_path + row.sid,exist_ok=True)
        #mesh_local.export(save_path + row.folder + '__' + row.sid + '__tooth' + str(row.teeth_id) + '__local.stl')
        mesh_world.export(save_path + row.sid+'/'+str(row.teeth_id) + '__world.stl')
    else:
        if not os.path.exists(save_path + row.sid):
            os.makedirs(save_path + row.sid,exist_ok=True)
        os.system('copy  "%s" "%s"' % ( mesh_file, save_path+ row.sid) )
    if not os.path.exists(save_path+ row.sid+'\Location.txt'):
        print('not exit and copy')
        axis_file = os.path.dirname(mesh_file) + '\Location.txt'
        os.system('copy  "%s" "%s"' % ( axis_file, save_path+ row.sid) )
    
    ##
    # np.save(save_path + row.folder + '__' + row.sid + '__tooth' + str(row.teeth_id) + '__rotation_local2world.npy', 
    #         rotation_local2world) 
    # np.save(save_path + row.folder + '__' + row.sid + '__tooth' + str(row.teeth_id) + '__rotation_world2local.npy', 
    #         rotation_world2local)

        # opt.write('%s\t%s\t%s\n' % (row.folder, row.sid, row.teeth_id))
    #break
# opt.close()
