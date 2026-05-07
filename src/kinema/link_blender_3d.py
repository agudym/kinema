# Robot Kinematics Playground (Python Library)
# Author: anton.gudym@gmail.com (Anton Gudym)

from typing import Optional, Union, List, Dict, Tuple
import os
import bpy

import numpy as np

import mathutils
from .elementary_transforms import ET, SE3

from kinema import LinkKinematic

def create_hexapod(mesh_root: str) :
    # Main dimensions of the main parts - platform, hinges, pistons, etc.
    base_height = 0.25
    hinge_height = 0.75
    piston_height = 3
    hinge_ball_height = 0.5

    # Polygon meshes of the corresponding parts
    platform_path = os.path.join(mesh_root, "Platform_R_10_dZ_0.25.dae")
    hinge_path = os.path.join(mesh_root, "Hinge_OX_dZ_0.75.dae")
    sleeve_path = os.path.join(mesh_root, "Sleeve_OZ_3.dae")
    piston_path = os.path.join(mesh_root, "Piston_OZ_3.dae")
    ball_path = os.path.join(mesh_root, "Ball_dZ_0.5.dae")
    hinge_ball_path = os.path.join(mesh_root, "HingeBall_OX_dZ_0.5.dae")

    # Function to create a prismatic joint
    def create_piston_chain(link_parents, ets_parents):
        l = LinkBlenderDrawing3D(hinge_path, link_parents=link_parents,
                                ets_l2ps=ets_parents )
        l = LinkBlenderDrawing3D(None, link_parents=l, #q0_l2ps=np.pi/4,
                                ets_l2ps=ET.tz(hinge_height) * ET.Rx())
        l = LinkBlenderDrawing3D(None, link_parents=l, #q0_l2ps=np.pi/4,
                                ets_l2ps=ET.Rz(np.pi/2) * ET.Rx())
        l = LinkBlenderDrawing3D(hinge_path, link_parents=l,
                                ets_l2ps=ET.tz(hinge_height) * ET.Rx(np.pi))
        l = LinkBlenderDrawing3D(sleeve_path, link_parents=l,
                                ets_l2ps=ET.Rx(np.pi))
        l = LinkBlenderDrawing3D(piston_path, link_parents=l, #q0_l2ps=0.9 * piston_height,
                                ets_l2ps=ET.tz())#
        l = LinkBlenderDrawing3D(ball_path, link_parents=l,
                                ets_l2ps=ET.tz(piston_height + hinge_ball_height) * ET.Rx(np.pi) )
        l = LinkBlenderDrawing3D(hinge_ball_path, link_parents=l, #q0_l2ps=[[np.pi/4, np.pi/4, 0],],
                                ets_l2ps=ET.tz(hinge_ball_height/2) * ET.Rx() * ET.Ry() * ET.Rz() * ET.tz(-hinge_ball_height/2) )
        return l

    # Function to define one platform and attached hinges (section)
    def create_hex_section(link_platform: LinkBlenderDrawing3D, num_piston_chains: int):
        pistons_placement_radius = 4
        angular_step = 2 * np.pi / num_piston_chains

        links_out = []
        for i in range(num_piston_chains):
            ets = ET.tz(base_height) * ET.Rz(angular_step * i) * ET.ty(pistons_placement_radius)
            links_out.append(create_piston_chain(link_platform, ets))
        return links_out

    # Creation of the complete mechanism - connection of several sections
    num_sections = 3
    # Number of prismatic joints in a section
    num_piston_chains = 6
    link_platform = LinkBlenderDrawing3D(platform_path)
    for section_id in range(num_sections):
        links_out = create_hex_section(link_platform, num_piston_chains)

        ets_platform_to_chains = []
        for i in range(num_piston_chains):
            ets_parent_to_world = links_out[i].get_orientation()
            if i == 0:
                # Use first chain for reference orientation(Z-height)
                ets_platform_to_world = ET.tz(ets_parent_to_world.mat[2,3])
            
            ets_platform_to_chains.append(
                ets_parent_to_world.inv() * ets_platform_to_world)

        link_platform = LinkBlenderDrawing3D(platform_path, link_parents=links_out,
                                            ets_l2ps=ets_platform_to_chains)

    link_platform.init_transform()

    return link_platform

def create_panda(mesh_root:str) :
    # Panda's kinematics, step-by-step
    panda_meshes = [
        ("link0.dae", None ),
        ("link1.dae", ET.tz(0.333) * ET.Rz()),
        ("link2.dae", ET.Rx(-np.pi/2) * ET.Rz()),
        ("link3.dae", ET.ty(-0.316) * ET.Rx(np.pi/2) * ET.Rz()),
        ("link4.dae", ET.tx(0.0825) * ET.Rx(np.pi/2) * ET.Rz()),
        ("link5.dae", ET.tx(-0.0825) * ET.ty(0.384) * ET.Rx(-np.pi/2) * ET.Rz()),
        ("link6.dae", ET.Rx(np.pi/2) * ET.Rz()),
        ("link7.dae", ET.tx(0.088) * ET.Rx(np.pi/2) * ET.Rz()),
        #   "hand.dae"
        #   "finger.dae",
        ]

    panda_links = []
    for i, (panda_mesh_path, ets_link2parent) in enumerate(panda_meshes):
        panda_links.append(
            LinkBlenderDrawing3D(
                os.path.join(mesh_root, panda_mesh_path),
                link_parents=panda_links[-1] if i > 0 else None,
                ets_l2ps=ets_link2parent))
    panda_links[-1].init_transform()
    return panda_links[-1]

class LinkBlenderDrawing3D(LinkKinematic):
    def __init__(self,
                mesh_filepath: Optional[str]=None,
                *args,
                **kwargs):
        
        super().__init__(*args, **kwargs)
        
        if mesh_filepath is not None:
            self.blender_mesh = BlenderObject(mesh_filepath, obj_name=self._name)
            # Apply transformation
            #self.blender_mesh.set_transform(self.get_orientation())
        else:
            self.blender_mesh =  None

    def init_transform(self):
        self.correct_initial_configuration()
        self._update_transform(self.collect_q0())

    def update_transform(self, link_name:str, T_goal: SE3, keyframe_animation=False, frame=None):
        q0 = None
        for link in self._list_all_links():
            if link.blender_mesh is None or link.blender_mesh.name != link_name:
                continue

            link.add_constraint(T_goal, reset=True)
            q0 = link.solve().q
            break
        if q0 is not None:
            self.reset_q0(q0)
            self._update_transform(q0, keyframe_animation, frame)
        else:
            raise ValueError(f"Could not find Link with name `{link_name}`")

    def _update_transform(self, _q, keyframe_animation=False, frame=None):
        for link in self._list_all_links():
            if link.blender_mesh is not None:
                if frame is not None:
                    bpy.context.scene.frame_set(frame)
                link.blender_mesh.set_transform(link.get_orientation(_q), keyframe_animation=keyframe_animation)

class BlenderObject():
    """ Handling object in blender """

    # Total amount of objects - to get a unique name
    obj_counter = 0

    def __init__(self, mesh_filepath: str, obj_name: Optional[str]=None):
        """
            Parameters
            ----------
            import_filepath : str
                path to the mesh that will be imported
        """
        BlenderObject.obj_counter += 1
        
        # Ensure path is absolute and uses forward slashes (safer for Blender operators)
        mesh_filepath = os.path.abspath(mesh_filepath).replace("\\", "/")
        self.original_filepath = mesh_filepath

        if not os.path.exists(mesh_filepath):
            raise FileNotFoundError(f"Mesh file not found at: {mesh_filepath}")

        self.name, self.ext = os.path.splitext(os.path.split(mesh_filepath)[1])
        if obj_name is not None:
            self.name = obj_name
        self.name = f"{self.name}_{BlenderObject.obj_counter}"
        
        if self.ext.lower() == ".dae":
            bpy.ops.wm.collada_import(filepath=mesh_filepath,
                                        auto_connect = True, 
                                        find_chains = True, 
                                        fix_orientation = True)
        elif self.ext.lower() == ".obj":
            bpy.ops.import_mesh.obj(filepath=mesh_filepath)
        elif self.ext.lower() == ".stl":
            bpy.ops.import_mesh.stl(filepath=mesh_filepath)

        # Group all nodes 
        if len(bpy.context.selected_objects) > 0:
            if len(bpy.context.selected_objects) > 1:
                bpy.ops.object.join()
            self.blender_object = bpy.context.selected_objects[0]

            self.blender_object.name = self.name
            self.blender_object.data.name = self.name

            self._apply_transform()
        else:
            raise RuntimeError(f"Failed to import mesh from {mesh_filepath}. No objects were imported.")
        
    def get_object(self):
        return self.blender_object
    
    def get_object_name(self):
        return self.name

    def set_transform(self, M, keyframe_animation=False):
        """ M is SE3 matrix of numpy.ndarray or mathutils.Matrix types, matrix_world_new = M
            keyframe_animation - insert keyframe for animation
        """ 
        self.blender_object.matrix_world = BlenderObject._fix_type(M)
        if keyframe_animation:
            self.blender_object.keyframe_insert(data_path="location", index=-1)
            self.blender_object.keyframe_insert(data_path="rotation_euler", index=-1)
        
    def add_transform(self, M):
        """ M is SE3 matrix of numpy.ndarray or mathutils.Matrix types, 
            matrix_world_new = M @ matrix_world
        """ 
        self.blender_object.matrix_world = BlenderObject._fix_type(M) @ self.blender_object.matrix_world

    @staticmethod
    def _fix_type(M):
        if isinstance(M, mathutils.Matrix):
            return M
        elif isinstance(M, np.ndarray):
            return mathutils.Matrix(M)
        elif isinstance(M, SE3):
            return mathutils.Matrix(M.mat)
        else:
            raise ValueError(f"Can not convert object of type {type(M)} into mathutils.Matrix")
        
    def _apply_transform(self):
        mb = self.blender_object.matrix_basis
        if hasattr(self.blender_object.data, "transform"):
            self.blender_object.data.transform(mb)
        for c in self.blender_object.children:
            c.matrix_local = mb @ c.matrix_local
        self.blender_object.matrix_basis.identity()
