# Generating animated 3D scenes in Blender
# Robot Kinematics Playground (Python Library)
# Author: anton.gudym@gmail.com (Anton Gudym)

# Blender3D python "backbone" package is required
import bpy

import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from kinema import create_hexapod, create_panda
from kinema.elementary_transforms import SE3

if __name__ == "__main__":
    root_path = os.path.split(__file__)[0]

    for model, create_robot in zip(("panda", "hexapod"), (create_panda, create_hexapod)):
        # Clean the BPY scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        mesh_root = os.path.abspath(f"{root_path}/../assets/meshes/{model}/")
        
        link_last = create_robot(mesh_root)

        target_object = link_last.blender_mesh.get_object()
        T_initial = np.array(target_object.matrix_world)

        if model == "hexapod":
            # Trajectory for animation: for hexapod we should rotate the last platform +-90 deg
            angles = np.linspace(-np.pi/2, np.pi/2, 20)
            frames = range(1, 21)
            
            for frame, angle in tqdm(zip(frames, angles), total=len(frames), desc=f"Generating {model} animations"):
                dT = np.identity(4)
                dT[:3,:3] = R.from_euler('z', angle).as_matrix()
                T_goal = T_initial @ dT
                
                link_last.update_transform(target_object.name, SE3(T_goal), keyframe_animation=True, frame=frame)
                
        elif model == "panda":
            # Trajectory for animation: for panda - already in home position lets move the last link closer to the origin (0,0,0)
            translations = np.linspace(0, -0.3, 20)
            frames = range(1, 21)
            
            for frame, translation in tqdm(zip(frames, translations), total=len(frames), desc=f"Generating {model} animations"):
                T_goal = T_initial.copy()
                T_goal[2, 3] += translation
                
                link_last.update_transform(target_object.name, SE3(T_goal), keyframe_animation=True, frame=frame)

        # Save the created scene into a .blend file
        output_path = os.path.abspath(f"kinematics_{model}.blend").replace("\\", "/")
        bpy.ops.wm.save_as_mainfile(filepath=output_path, compress=True)
        print(f"Scene successfully saved to: {output_path}")
