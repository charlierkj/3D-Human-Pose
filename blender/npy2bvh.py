import os
import numpy as np
import bpy

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

    def __init__(self):
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
        
    def convert(self, npy_path, write_path):
        joints_3d_np = np.load(npy_path)
        joints_3d_np = joints_3d_np / 100 # cm -> m
        num_frames = len(joints_3d_np)
        for frame in range(num_frames):
            skeleton_scaled = self.scale_skeleton(joints_3d_np[frame, ...])

            bpy.context.scene.frame_set(frame)
            bpy.data.scenes['Scene'].frame_end = frame + 1
            for obj in self.empties:
                if obj.name in self.joints_name:
                    joint_idx = self.joints_name.index(obj.name)
                    obj.location = skeleton_scaled[joint_idx, :]                    
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
            

if __name__ == "__main__":
    npy_path = 'C:/Users/charl/Work/hopkins_time/Research/learnable-triangulation-pytorch/results/test_01/preds/anim_000/joints_3d.npy'
    bvh_file = 'C:/Users/charl/Work/hopkins_time/Research/learnable-triangulation-pytorch/blender/bvh/anim_000.bvh'
    
    converter = NPY2BVH_Converter()
    converter.convert(npy_path, bvh_file)
