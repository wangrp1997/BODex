import bpy 
import trimesh 
import numpy as np
import os 
from tqdm import tqdm

def simplify(obj_path, save_path):
    tm_mesh = trimesh.load(obj_path, force='mesh')
    target_face = 5000
    decimate_ratio = target_face / tm_mesh.faces.shape[0]
    
    if decimate_ratio > 1:
        tm_mesh.export(save_path)
        return 
    
    new_mesh = bpy.data.meshes.new('object_mesh')
    new_mesh.from_pydata(tm_mesh.vertices.tolist(), [], tm_mesh.faces.tolist())
    new_mesh.update(calc_edges=True, calc_edges_loose=True)
    object_mesh = bpy.data.objects.new('object', new_mesh)
    bpy.context.collection.objects.link(object_mesh)
    bpy.context.view_layer.objects.active = object_mesh

    bpy.ops.object.modifier_add(type='DECIMATE') #Activates the Decimate modifier.
    bpy.context.object.modifiers["Decimate"].decimate_type = 'COLLAPSE' #Activates Collapse option.                        
    bpy.context.object.modifiers["Decimate"].ratio = decimate_ratio #Sets the ratio to be equal to ratioInput.
    bpy.context.object.modifiers["Decimate"].use_collapse_triangulate = True #Triangulate Option
    bpy.ops.object.modifier_apply(modifier="Decimate") #Applies the modifier.
            
    result = bpy.context.active_object.data

    vertices = [v.co for v in result.vertices]
    faces = [f.vertices for f in result.polygons]

    # Convert to NumPy arrays
    vertices_np = np.array(vertices)
    faces_np = np.array(faces)

    new_m = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
    new_m.export(save_path)
    
    print(tm_mesh.faces.shape[0], new_m.faces.shape[0], decimate_ratio)
    return 


def manifold(manifoldplus_path, obj_path, save_path):
    os.system(f'{manifoldplus_path} --input {obj_path} --output {save_path}')
    return 


obj_folder = 'src/curobo/content/assets/object/meshdata'
manifoldplus_path = 'thirdparty/ManifoldPlus/build/manifold'

id_lst = os.listdir(obj_folder)


for id in tqdm(id_lst):
    read_path = os.path.join(obj_folder, id, 'coacd', 'decomposed.obj')
    os.system(f"rm {read_path.replace('decomposed.obj', 'no_inter.obj')}")
    mani_path = read_path.replace('decomposed.obj', 'manifold.obj')
    manifold(manifoldplus_path, read_path, mani_path)
    simp_path = read_path.replace('decomposed.obj', 'simply.obj')
    simplify(mani_path, simp_path)