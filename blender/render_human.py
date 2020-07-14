import os, sys
import numpy as np
import bpy

blend_filepath = bpy.data.filepath
blend_dir = os.path.split(blend_filepath)[0]
print(blend_dir)
sys.path.append(blend_dir)

import tf_utils


def get_num_frames_from_bvh(bvh_file):
    f = open(bvh_file, 'r')
    line = f.readline()
    while line and line != '':
        if line.startswith('Frames:'):
            num_frames = int(line.split(' ')[1])
            f.close()
            return num_frames
        line = f.readline()
    print('Cannot find frame numbers in .bvh file')
    return 0
    

def render_human(bvh_file, viewpoint, target, img_size, save_folder):
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
    camera_loc, camera_rot_eulerXYZ = tf_utils.viewpoint_to_eulerXYZ(viewpoint, target)
    camera_obj.location = camera_loc
    #camera_obj.rotation_mode = 'XYZ' # Euler XYZ
    #camera_obj.rotation_euler = camera_rot_eulerXYZ
    camera_rot_quaternion = tf_utils.eulerXYZ_to_quaternion(camera_rot_eulerXYZ)
    camera_obj.rotation_mode = 'QUATERNION'
    camera_obj.rotation_quaternion = camera_rot_quaternion

    # render
    # set light
    light = bpy.data.lights[0]
    light.energy = 500
    # set resolution
    (bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y) = img_size
    bpy.context.scene.render.resolution_percentage = 100

    num_frames = get_num_frames_from_bvh(bvh_file)
    for frame_idx in range(2, num_frames): # from 2nd frame, 1st frame defective
        bpy.context.scene.frame_set(frame_idx)
        img_path = os.path.join(save_folder, '%06d.png' % (frame_idx - 1))
        bpy.data.scenes['Scene'].render.filepath = img_path
        bpy.ops.render.render(write_still=True) 


if __name__ == "__main__":

    if '--' not in sys.argv:
        bvh_file = 'C:/Users/charl/Work/hopkins_time/Research/learnable-triangulation-pytorch/blender/bvh/anim_100.bvh'
        save_folder = 'C:/Users/charl/Work/hopkins_time/Research/learnable-triangulation-pytorch/blender/bvh/anim_100'
    else:
        bvh_file = sys.argv[-2]
        save_folder = sys.argv[-1]

    render_human(bvh_file, (3, 45, 30), (0, 0, 1.3), (1280, 960), save_folder)
