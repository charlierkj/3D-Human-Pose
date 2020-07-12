import os
import subprocess

blender_path = 'C:/Users/charl/Work/Blender Foundation/Blender 2.83/blender.exe' # absolute path for blender.exe
current_dir = os.path.dirname(os.path.realpath(__file__))

def blender_convert(npy_path, bvh_file):

    blend_file = os.path.join(current_dir, 'csv_to_bvh.blend') # .blend file
    py_file = os.path.join(current_dir, 'npy2bvh.py')

    render_cmd = f'"{blender_path}" --background "{blend_file}" \
        --python "{py_file}" \
        -- "{npy_path}" "{bvh_file}"'
    print('Run blender command: %s' % render_cmd)
    subprocess.run(render_cmd)


def blender_render(bvh_file, save_folder):

    blend_file = os.path.join(current_dir, 'prototype_0522.blend') # .blend file
    py_file = os.path.join(current_dir, 'render_human.py')

    render_cmd = f'"{blender_path}" --background "{blend_file}" \
        --python "{py_file}" \
        -- "{bvh_file}" "{save_folder}"'
    print('Run blender command: %s' % render_cmd)
    subprocess.run(render_cmd)
