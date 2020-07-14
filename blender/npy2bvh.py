import os, sys
import numpy as np
import bpy


blend_filepath = bpy.data.filepath
blend_dir = os.path.split(blend_filepath)[0]
print(blend_dir)
sys.path.append(blend_dir)

import tf_utils


Joints_Pred = [
    "Ankle.R",
    "Knee.R",
    "Hip.R",
    "Hip.L",
    "Knee.L",
    "Ankle.L",
    "Hip.Center",
    "Spine", # no correspondence in bvh
    "Neck",
    "Head",
    "Wrist.R",
    "Elbow.R",
    "Shoulder.R",
    "Shoulder.L",
    "Elbow.L",
    "Wrist.L",
    "Nose"
    ]

Bones = [
    (6, 3),
    (3, 4),
    (4, 5),
    (6, 2),
    (2, 1),
    (1, 0),
    (6, 8),
    (8, 13),
    (13, 14),
    (14, 15),
    (8, 12),
    (12, 11),
    (11, 10),
    (8, 16),
    (16, 9)
    ]

class NPY2BVH_Converter(object):

    def __init__(self, centralize=False):
        self.joints_name = Joints_Pred
        self.bones = Bones
        
        objs = bpy.context.scene.objects
        self.empties = []
        for obj in objs:
            if obj.type == 'EMPTY':
                self.empties.append(obj)
        print(self.empties)
        
        # compute standard bone length initially
        self.bones_length = []
        for bone in self.bones:
            root_name = self.joints_name[bone[0]]
            child_name = self.joints_name[bone[1]]
            bone_length = (bpy.context.scene.objects[child_name].location -
                           bpy.context.scene.objects[root_name].location).length
            self.bones_length.append(bone_length)

        # compute standard facebone (ear, eye) initially
        self.facing_init = np.array([1, 0, 1])
        self.facebone = self.set_facebone() # dict
        self.nose_idx = self.joints_name.index('Nose') # nose will be used as reference

        self.centralize = centralize # whether initialliy place the skeleton at origin (X-Y plane) 
        
    def convert(self, npy_path, write_path):
        joints_3d_np = np.load(npy_path)
        
        if self.centralize:
            joints_3d_np = self.centralize_joints(joints_3d_np)

        # UE is not right-hand frame
        joints_3d_np[:, :, 1] = - joints_3d_np[:, :, 1]

        joints_3d_np = joints_3d_np / 100 # cm -> m
        num_frames = len(joints_3d_np)
        for frame in range(num_frames):
            skeleton = joints_3d_np[frame, ...]
            facing_curr = self.compute_facing_curr(skeleton) # facing direction
            rot = tf_utils.compute_rotation(self.facing_init, facing_curr)
            skeleton_scaled = self.scale_skeleton(skeleton)

            bpy.context.scene.frame_set(frame)
            bpy.data.scenes['Scene'].frame_end = frame + 1
            for obj in self.empties:
                if obj.name in self.joints_name:
                    joint_idx = self.joints_name.index(obj.name)
                    obj.location = skeleton_scaled[joint_idx, :]
                    obj.keyframe_insert(data_path="location", index=-1)

            for obj in self.empties:
                if obj.name in self.facebone.keys():
                    obj.location = skeleton_scaled[self.nose_idx, :] + rot @ self.facebone[obj.name]
                    obj.keyframe_insert(data_path="location", index=-1)

        bpy.data.objects['rig'].select_set(True)
        write_folder = os.path.split(write_path)[0]
        if write_folder != '':
            os.makedirs(write_folder, exist_ok=True)
        bpy.ops.export_anim.bvh(filepath=write_path)

    def scale_skeleton(self, skeleton):
        """skeleton: numpy array of size (num_joints, 3)."""
        skeleton_scaled = np.empty(shape=skeleton.shape)
        hip_idx = self.joints_name.index('Hip.Center') # hip/pelvis as the reference
        skeleton_scaled[hip_idx, :] = skeleton[hip_idx, :]
        for bone_idx, bone in enumerate(self.bones):
            vec = skeleton[bone[1], :] - skeleton[bone[0], :]
            vec_normalized = vec / np.linalg.norm(vec)
            skeleton_scaled[bone[1], :] = skeleton_scaled[bone[0], :] + \
                                          self.bones_length[bone_idx] * vec_normalized
        return skeleton_scaled

    def compute_facing_curr(self, skeleton):
        """compute current facing direction of predicted skeleton."""
        neck_idx = self.joints_name.index('Neck')
        nose_idx = self.joints_name.index('Nose')
        head_idx = self.joints_name.index('Head')
        facing = (skeleton[nose_idx, :] - skeleton[neck_idx, :]) + \
                 (skeleton[nose_idx, :] - skeleton[head_idx, :])
        facing = facing / np.linalg.norm(facing)
        return facing # unit vector

    def set_facebone(self):
        facebone = {'Ear.L': 0, 'Ear.R': 0, 'Eye.L': 0, 'Eye.R': 0}
        ref_loc = bpy.context.scene.objects['Nose'].location # Nose as reference
        for obj in self.empties:
            if obj.name in facebone.keys():
                joint_loc = obj.location
                facebone[obj.name] = np.array(joint_loc - ref_loc)
        return facebone

    def centralize_joints(self, joints_3d_np):
        hip_c_idx = self.joints_name.index('Hip.Center')
        offset = joints_3d_np[0, hip_c_idx, :] # first frame, hip center
        offset[2] = 0 # do not change Z value
        return joints_3d_np - offset
                  

if __name__ == "__main__":

    if '--' not in sys.argv:
        npy_path = 'C:/Users/charl/Work/hopkins_time/Research/learnable-triangulation-pytorch/results/test_01/preds/anim_100/joints_3d.npy'
        bvh_file = 'C:/Users/charl/Work/hopkins_time/Research/learnable-triangulation-pytorch/blender/bvh/anim_100.bvh'
    else:
        npy_path = sys.argv[-2]
        bvh_file = sys.argv[-1]
    
    converter = NPY2BVH_Converter(centralize=True)
    converter.convert(npy_path, bvh_file)
