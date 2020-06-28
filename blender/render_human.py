import os, sys, subprocess
import numpy as np
import bpy

blend_filepath = bpy.data.filepath
blend_dir = os.path.split(blend_filepath)[0]
print(blend_dir)
sys.path.append(blend_dir)

import tf_utils


def render_human(bvh_file, viewpoint, img_size, save_folder):
    """cam_pose: 3-tuple: (distance, azimuth, elevation)."""
    if save_folder != '':
        os.makedirs(save_folder, exist_ok=True)

    # add bvh file and set animation
    bpy.ops.import_anim.bvh(filepath=bvh_file, filter_glob='*.bvh', \
                            frame_start=1, axis_forward='X', axis_up='Z')
    bvh_filename = os.path.basename(bvh_file)
    anim_name = os.path.splitext(bvh_filename)[0]
    skeleton_obj = bpy.data.objects['originalPose']
    skeleton_obj.animation_data.action = bpy.data.actions.get(anim_name)
    skeleton_obj.rotation_euler = [-3.1416/2, 0, 3.1416/2] # need to modify
    bpy.data.objects[anim_name].select_set(True)
    bpy.ops.object.delete()

    bpy.context.scene.frame_set(0)

    # place camera
    camera_obj = bpy.data.objects['Camera']
    camera_loc, camera_rot_eulerXYZ = tf_utils.viewpoint_to_eulerXYZ(viewpoint)
    camera_obj.location = camera_loc
    #camera_obj.rotation_mode = 'XYZ' # Euler XYZ
    #camera_obj.rotation_euler = camera_rot_eulerXYZ
    camera_rot_quaternion = tf_utils.eulerXYZ_to_quaternion(camera_rot_eulerXYZ)
    camera_obj.rotation_mode = 'QUATERNION'
    camera_obj.rotation_quaternion = camera_rot_quaternion

    # render
    # set resolution
    (bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y) = img_size
    
    for frame_idx in range(2, 61): # hardcoded, need later modification
        bpy.context.scene.frame_set(frame_idx)
        img_path = os.path.join(save_folder, '%06d.png' % (frame_idx - 1))
        bpy.data.scenes['Scene'].render.filepath = img_path
        bpy.ops.render.render(write_still=True)       
    
    
def blender_cmd(bvh_file, cam_pose, save_folder):

    blender_path = 'C:/Users/charl/Work/Blender Foundation/Blender 2.83/blender.exe' # absolute path for blender.exe
    blend_file = 'prototype_0522.blend' # .blend file
    py_file = 'npy2bvh.py'

    render_cmd = f'"{blender_path}" "{blend_file}" --python "{py_file}"'
    print('Run blender command: %s' % render_cmd)
    subprocess.run(render_cmd)


if __name__ == "__main__":
    bvh_file = 'C:/Users/charl/Work/hopkins_time/Research/learnable-triangulation-pytorch/blender/bvh/anim_100.bvh'
    save_folder = 'C:/Users/charl/Work/hopkins_time/Research/learnable-triangulation-pytorch/blender/bvh/anim_100'

    render_human(bvh_file, (5, 45, 60), (640, 480), save_folder)
