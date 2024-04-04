#!/usr/bin/python
# -*- coding: utf-8-*-
'''
this python file is used to generate the data for the primeshapes dataset using blender.
'''

import os, sys
import bpy
import numpy as np
from mathutils import Matrix, Vector
import subprocess
import sys
import os
os.system("which pip")
import pandas
import pickle

argv = sys.argv
#argv = argv[argv.index("--") + 1:]

homepath = '/home/virajs/sensei-fs-symlink/tenants/Sensei-AdobeResearchTeam/blender_datagen/'

with open(homepath + "material_id.pkl", "rb") as handle:
    material_id = pickle.load(handle)

#print(argv[1])

# files = sorted(os.listdir("/home/prafulls/sensei-fs-symlink/users/prafulls/latest_renderings/"))
# if str(int(argv[1])).zfill(6) in files:
#     exit(0)

df = pandas.read_csv(homepath + "stochastics_allcategories.csv", header=None)
valid_materials = df[0].to_list()

def remove_all_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def remove_all_materials():
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)


def remove_all_nodes(nodes):
    for n in nodes:
        nodes.remove(n)


def new_plane(name, loc, sz, rot=(0, 0, 0), scale=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(
        size=sz,
        calc_uvs=True,
        enter_editmode=False,
        align='WORLD',
        location=loc,
        rotation=rot,
        scale=scale)

    current_name = bpy.context.selected_objects[0].name

    bpy.context.selected_objects[0].scale = scale
    plane = bpy.data.objects[current_name]
    plane.name = name
    plane.data.name = name + '_mesh'
    return plane


def new_sphere(name, loc, radius):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=512,
                                         ring_count=512,
                                         radius=radius,
                                         location=loc)
    current_name = bpy.context.selected_objects[0].name
    sphere = bpy.data.objects[current_name]

    sphere.name = name
    sphere.data.name = name + '_mesh'
    return sphere


def new_cuboid(name, loc, size, rot):
    bpy.ops.mesh.primitive_cube_add(size=size,
                                    enter_editmode=False, align='WORLD',
                                    location=loc,
                                    rotation=rot,
                                    scale=(1, 1, 1))
    current_name = bpy.context.selected_objects[0].name
    cuboid = bpy.data.objects[current_name]
    cuboid.name = name
    cuboid.data.name = name + '_mesh'
    return cuboid


def new_cylinder(name, loc, radius, depth, rot):
    bpy.ops.mesh.primitive_cylinder_add(vertices=512,
                                        radius=radius,
                                        depth=depth,
                                        enter_editmode=False,
                                        align='WORLD',
                                        location=loc,
                                        rotation=rot,
                                        scale=(1, 1, 1))
    current_name = bpy.context.selected_objects[0].name
    cylinder = bpy.data.objects[current_name]

    cylinder.name = name
    cylinder.data.name = name + '_mesh'
    return cylinder


def new_cone(name, loc, rad1, rad2, depth, rot):
    bpy.ops.mesh.primitive_cone_add(vertices=512,
                                    radius1=rad1,
                                    radius2=rad2,
                                    depth=depth,
                                    enter_editmode=False,
                                    align='WORLD',
                                    location=loc,
                                    # rotation=(0,-np.pi,0),
                                    rotation=rot,
                                    scale=(1, 1, 1))
    current_name = bpy.context.selected_objects[0].name
    cone = bpy.data.objects[current_name]

    cone.name = name
    cone.data.name = name + '_mesh'
    return cone


def new_torus(name, loc, major_radius, minor_radius, abso_major_rad, abso_minor_rad, rot):
    bpy.ops.mesh.primitive_torus_add(major_segments=256,
                                     minor_segments=256,
                                     align='WORLD',
                                     location=loc,
                                     rotation=rot,
                                     major_radius=major_radius,
                                     minor_radius=minor_radius,
                                     abso_major_rad=abso_major_rad,
                                     abso_minor_rad=abso_minor_rad)
    current_name = bpy.context.selected_objects[0].name
    torus = bpy.data.objects[current_name]

    torus.name = name
    torus.data.name = name + '_mesh'
    return torus


def new_light(name, light_type, mat_world, size=0.1, energy=400):
    bpy.ops.object.light_add(type=light_type, radius=size, align='WORLD', location=mat_world[:3, -1],
                             rotation=(0, 0, 0), scale=(1, 1, 1))
    current_name = bpy.context.selected_objects[0].name
    light = bpy.data.objects[current_name]
    light.name = name
    light.data.name = name
    light.data.energy = energy
    print(bpy.data.objects[name].matrix_world, mat_world[:3, :3])
    bpy.data.objects[name].matrix_world = Matrix(mat_world)
    return light


def add_object(object_type, name):
    if object_type == 0:
        size = np.random.uniform(0.5, 2)
        loc = sample_location(-floor_length/2 + size/2, floor_length/2-size/2, 4)
        rot = np.random.uniform(-np.pi/2, np.pi/2, (3,))
        loc[-1] = size/2
        return new_cuboid(name, loc, size, rot)
    elif object_type == 1:
        radius = np.random.uniform(0.5, 1.5)
        loc = sample_location(-floor_length/2 + radius, floor_length/2 - radius, 3)
        loc[-1] = radius
        return new_sphere(name, loc, radius)
    elif object_type == 2:
        radius = np.random.uniform(0.5, 1.5)
        depth = np.random.uniform(0.5, 2)
        rot = np.random.uniform(-np.pi/4, np.pi/4, (3,))
        loc = sample_location(-floor_length/2 + radius, floor_length/2 - radius, 4)
        loc[-1] = depth/2
        return new_cylinder(name, loc, radius, depth, rot)
    elif object_type == 3:
        rad1, rad2 = np.random.uniform(1., 2.), 0
        depth = np.random.uniform(rad1, 2.)
        rot = np.random.uniform(-np.pi/3, np.pi/3, (3,)) #- np.pi
        rot[1] = 0
        loc = sample_location(-floor_length/2 + rad1, floor_length/2 - rad1, 4)
        loc[-1] = depth/2
        return new_cone(name, loc, rad1, rad2, depth, rot)
    elif object_type == 4:
        radius1, radius2 = np.random.uniform(0.5, 1.5, (2,))
        major_radius, abso_major_rad = min(radius1, radius2), max(radius1, radius2)
        radius1, radius2 = np.random.uniform(0.15, 0.5, (2,))
        minor_radius, abso_minor_rad = min(radius1, radius2), max(radius1, radius2)
        rot = np.random.uniform(-np.pi/2, np.pi/2, (3,))
        rot[1] = 0
        loc = sample_location(-floor_length/2 + abso_major_rad, floor_length/2 - abso_major_rad, abso_major_rad*1.5)
        loc[-1] = abso_major_rad/2
        return new_torus(name, loc, major_radius, minor_radius, abso_major_rad, abso_minor_rad, rot)

def get_shaderimg_node(img_path, name, mat_nodes):
    tex_image = mat_nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(img_path)
    tex_image.name = name
    mat_nodes[name].label = name
    return tex_image

def generatePBR_material(path, name):
    material_files = [os.path.join(path, x) for x in os.listdir(path) if '.png' in x]
    color_img_path = [x for x in material_files if 'basecolor' in x.lower()][0]
    roughness_img_path = [x for x in material_files if 'roughness' in x.lower()][0]
    normal_img_path = [x for x in material_files if 'normal' in x.lower()][0]
    displacement_img_path = [x for x in material_files if 'height' in x.lower()][0]
    metallic_img_path = [x for x in material_files if 'metallic' in x.lower()]
    #
    if len(metallic_img_path) == 1:
        metallic_img_path = metallic_img_path[0]
    else:
        metallic_img_path = None
    #
    specular_img_path = [x for x in material_files if 'specular' in x.lower()]
    if len(specular_img_path) == 1:
        specular_img_path = specular_img_path[0]
    else:
        specular_img_path = None
    #   
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.cycles.displacement_method = "BOTH"
    mat_node_tree = mat.node_tree
    mat_nodes = mat_node_tree.nodes
    shader = mat_nodes['Principled BSDF']
    #    shader
    mat_output = mat_nodes['Material Output']
    #
    texCoord = mat_nodes.new('ShaderNodeTexCoord')
    texCoord.location = texCoord.location - Vector((800, 0))
    texMapping = mat_nodes.new('ShaderNodeMapping')
    texMapping.location = texMapping.location - Vector((600, 0))
    mat_node_tree.links.new(texCoord.outputs[2], texMapping.inputs[0])
    #
    color_node = get_shaderimg_node(color_img_path, 'Base Color', mat_nodes)
    color_node.location = Vector((-400, 300))
    mat_node_tree.links.new(shader.inputs[0], color_node.outputs[0])
    mat_node_tree.links.new(texMapping.outputs[0], color_node.inputs[0])
    #
    if metallic_img_path is not None:
        metallic_node = get_shaderimg_node(roughness_img_path, 'Metallic', mat_nodes)
        metallic_node.image.colorspace_settings.name = "Non-Color"
        metallic_node.location = Vector((-400, 0))
        mat_node_tree.links.new(shader.inputs[6], metallic_node.outputs[0])
        mat_node_tree.links.new(texMapping.outputs[0], metallic_node.inputs[0])
    
    if specular_img_path is not None:
        specular_node = get_shaderimg_node(specular_img_path, 'Specular', mat_nodes)
        specular_node.image.colorspace_settings.name = "Non-Color"
        specular_node.location = Vector((-400, -300))
        mat_node_tree.links.new(shader.inputs[7], specular_node.outputs[0])
        mat_node_tree.links.new(texMapping.outputs[0], specular_node.inputs[0])    
    #
    roughness_node = get_shaderimg_node(roughness_img_path, 'Roughness', mat_nodes)
    roughness_node.image.colorspace_settings.name = "Non-Color"
    roughness_node.location = Vector((-400, -600))
    mat_node_tree.links.new(shader.inputs[9], roughness_node.outputs[0])
    mat_node_tree.links.new(texMapping.outputs[0], roughness_node.inputs[0])
    #
    normal_node = get_shaderimg_node(normal_img_path, 'Normal', mat_nodes)
    normal_node.image.colorspace_settings.name = "Non-Color"
    normal_node.location = Vector((-400, -900))
    normal_map = mat_nodes.new('ShaderNodeNormalMap')
    normal_map.location = Vector((-175, -900))
    mat_node_tree.links.new(texMapping.outputs[0], normal_node.inputs[0])
    mat_node_tree.links.new(normal_node.outputs[0], normal_map.inputs[1])
    mat_node_tree.links.new(normal_map.outputs[0], shader.inputs[22])
    #
    displacement_node = get_shaderimg_node(displacement_img_path, 'Displacement', mat_nodes)
    displacement_node.image.colorspace_settings.name = "Non-Color"
    displacement_node.location = Vector((-400, -1200))
    disp = mat_nodes.new('ShaderNodeDisplacement')
    disp.inputs[2].default_value = np.random.uniform(0.15, 0.25)
    disp.location = Vector((-175, -900))
    mat_node_tree.links.new(texMapping.outputs[0], displacement_node.inputs[0])
    mat_node_tree.links.new(displacement_node.outputs[0], disp.inputs[0])
    mat_node_tree.links.new(disp.outputs[0], mat_output.inputs[2])
    return mat

def get_material(folder_name):
    all_materials = []
    for d in os.listdir(TEXTURE_DIR):
        if not os.path.isdir(os.path.join(TEXTURE_DIR, d)):
            continue
        if folder_name in d:
            tex_dirs = [os.path.join(TEXTURE_DIR, d, i) for i in sorted(os.listdir(os.path.join(TEXTURE_DIR, d)))]
            for x in tex_dirs:
                # print(x[x.rfind("/")+1:])
                if x[x.rfind("/")+1:] + ".sbsar" in valid_materials:
                    if os.path.isdir(x):
                        for i in range(4):
                            if os.path.exists(os.path.join(x, str(i))) and os.path.isdir(os.path.join(x, str(i))):
                                all_materials.append(os.path.join(x, str(i)))
    print(folder_name, len(all_materials))
    random_material_idx = np.random.randint(len(all_materials))
    material_path = all_materials[random_material_idx]
    return material_path


def makeLookAt(position, target, up):
    forward = np.subtract(target, position)
    forward = np.divide(forward, np.linalg.norm(forward))
    right = np.cross(forward, up)
    # if forward and up vectors are parallel, right vector is zero;
    #   fix by perturbing up vector a bit
    if np.linalg.norm(right) < 0.001:
        epsilon = np.array([0.001, 0, 0])
        right = np.cross(forward, up + epsilon)
    right = np.divide(right, np.linalg.norm(right))
    up = np.cross(right, forward)
    up = np.divide(up, np.linalg.norm(up))
    return np.array([[right[0], up[0], -forward[0], position[0]],
                     [right[1], up[1], -forward[1], position[1]],
                     [right[2], up[2], -forward[2], position[2]],
                     [0, 0, 0, 1]])


def acceptable_location(location, dist_thresh):
    for obj in bpy.data.objects:
        if 'object' not in obj.name:
            continue
        dist_obj = np.linalg.norm(location - obj.location)
        if dist_obj <= dist_thresh:
            return False
    return True

def get_random_location(min_loc, max_loc):
    location = np.random.uniform(min_loc, max_loc, (3,))
    location[-1] = np.random.uniform(1.5, 6)
    return location

def sample_location(min_loc, max_loc, dist_threhold):
    location = get_random_location(min_loc, max_loc)
    while not acceptable_location(location, dist_threhold):
        location = get_random_location(min_loc, max_loc)
    return location



dist_thresh = 2.
TEXTURE_DIR = homepath + '/valentin_materials/'
np.random.seed(int(argv[1]))

# Removing all objects in the scene and setting the global parameters.
remove_all_objects()
remove_all_materials()
floor_length = 10
wall_height = 6
wall_length = 10

# Light parameter ranges
NUM_LIGHTS = np.random.randint(3, 6)
light_size_min = 0.1
light_size_max = 1
light_energy_min = 300
light_energy_max = 600
texture_types = [x for x in sorted(os.listdir(TEXTURE_DIR)) if os.path.isdir(os.path.join(TEXTURE_DIR, x))]
print(texture_types)
texture_types.remove("Atlas")
texture_types.remove("Ceramic")
texture_types.remove("Foliage")
print(texture_types)

floor_mat_types = texture_types  # ['clay', 'stone']
wall_mat_types = texture_types  # ['clay', 'stone']

materials_used = []

# Create floor as a plane
floor = new_plane('floor', loc=(0, 0, 0), sz=2, rot=(0, 0, 0), scale=(floor_length / 2 + 1, floor_length / 2 + 1, 1))
floor_type_idx = np.random.choice(floor_mat_types)
floor_path = get_material(floor_type_idx)
print(floor_path)
floor_mat = generatePBR_material(floor_path, 'floor_mat')
mat_name = floor_path.split("/")[-2]
mat_variant = floor_path.split("/")[-1]
floor_mat.pass_index = material_id[mat_name + "_" + mat_variant]
# cur_material_id += 1
floor.data.materials.append(floor_mat)

# Create wall as plane objects
wall_1 = new_plane('wall_1', loc=(0, -floor_length // 2, wall_height / 2), sz=2, rot=(np.pi / 2, 0, 0),
                   scale=(wall_length / 2 + 1, wall_height / 2 + 1, 1))
wall_2 = new_plane('wall_2', loc=(0, floor_length // 2, wall_height / 2), sz=2, rot=(-np.pi / 2, 0, 0),
                   scale=(wall_length / 2 + 1, wall_height / 2 + 1, 1))
wall_3 = new_plane('wall_3', loc=(-floor_length // 2, 0, wall_height / 2), sz=2, rot=(-np.pi / 2, 0, np.pi / 2),
                   scale=(wall_length / 2 + 1, wall_height / 2 + 1, 1))
wall_4 = new_plane('wall_4', loc=(floor_length // 2, 0, wall_height / 2), sz=2, rot=(np.pi / 2, 0, np.pi / 2),
                   scale=(wall_length / 2 + 1, wall_height / 2 + 1, 1))

# Wall material
wall_type_idx = np.random.choice(wall_mat_types)
wall_path = get_material(wall_type_idx)
wall_mat = generatePBR_material(wall_path, 'wall_mat')
mat_name = wall_path.split("/")[-2]
mat_variant = wall_path.split("/")[-1]
wall_mat.pass_index = material_id[mat_name + "_" + mat_variant]
# cur_material_id += 1

# Assigning materials to wall
wall_1.data.materials.append(wall_mat)
wall_2.data.materials.append(wall_mat)
wall_3.data.materials.append(wall_mat)
wall_4.data.materials.append(wall_mat)

# Placing the objects
NUM_OBJECTS = 6
objects = []

for i in range(NUM_OBJECTS):
    object_idx = np.random.randint(5)
    objects.append(add_object(object_idx, 'object_' + str(i).zfill(2)))
    obj_type_idx = np.random.choice(texture_types)
    obj_path = get_material(obj_type_idx)
    #    while obj_path in materials_used:
    #        obj_type_idx = np.random.choice(texture_types)
    #        obj_path = get_material(obj_type_idx)
    materials_used.append(obj_path)
    obj_mat = generatePBR_material(obj_path, 'obj_mat_' + str(i).zfill(2))
    mat_name = obj_path.split("/")[-2]
    mat_variant = obj_path.split("/")[-1]
    obj_mat.pass_index = material_id[mat_name + "_" + mat_variant]
    objects[-1].data.materials.append(obj_mat)

# Smart UV projection
# Get all objects in selection
selection = bpy.context.selected_objects

# Get the active object
active_object = bpy.context.active_object

# Deselect all objects
bpy.ops.object.select_all(action='DESELECT')

for obj in selection:
    # Select each object
    obj.select_set(True)
    # Make it actives
    bpy.context.view_layer.objects.active = obj
    # Toggle into Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    # Select the geometry
    bpy.ops.mesh.select_all(action='SELECT')
    # Call the smart project operator
    # bpy.ops.uv.align(axis='ALIGN_AUTO')
    bpy.ops.uv.smart_project()
    # Toggle out of Edit Mode
    bpy.ops.object.mode_set(mode='OBJECT')
    # Deselect the object
    obj.select_set(False)
    
# Add Lighting to the room
lights = []
for i in range(NUM_LIGHTS):
    light_size = np.random.uniform(light_size_min, light_size_max)
    energy = np.random.randint(light_energy_min, light_energy_max)
    x, y, z = np.random.uniform(-floor_length / 2 + light_size / 2, floor_length / 2 - light_size / 2, (3,))
    rand_val = np.random.randint(5)
    y = np.random.uniform(2, wall_height)
    rand_obj = objects[np.random.randint(len(objects))]
    if len(lights) == 0:
        rand_val = 0
    if rand_val == 0:
        loc = (x, z, np.random.uniform(4, 6))
    elif rand_val == 1:
        loc = (x, -4.9, y)
    elif rand_val == 2:
        loc = (x, 4.9, y)
    elif rand_val == 3:
        loc = (-4.9, z, y)
    elif rand_val == 4:
        loc = (4.9, z, y)
    mat_world = makeLookAt(Vector(loc), rand_obj.location, np.array([0, 0, 1]))
    mat_world[:3, -1] = Vector(loc)
    print(mat_world)
    light_cur = new_light('lamp_' + str(i), 'AREA', mat_world=mat_world, size=light_size, energy=energy)
    lights.append(light_cur)

scene = bpy.context.scene
renderer = scene.render

# Setup renderer
renderer.engine = 'CYCLES'
scene.cycles.sampls = 256
renderer.resolution_x = 1024
renderer.resolution_y = 1024
renderer.resolution_percentage = 100
scene.frame_start = 0
scene.frame_end = 0

# Setup scene passes to render
scene.view_layers['ViewLayer'].use_pass_z = True
scene.view_layers['ViewLayer'].use_pass_normal = True
scene.view_layers['ViewLayer'].use_pass_material_index = True

# Setup output
scene.use_nodes = True
scene_node_tree = scene.node_tree
scene_nodes = scene_node_tree.nodes

remove_all_nodes(scene_nodes)
render_layers_node = scene_nodes.new('CompositorNodeRLayers')
composite_node = scene_nodes.new('CompositorNodeComposite')
composite_node.location = Vector((300, 0))
scene_node_tree.links.new(render_layers_node.outputs[0], composite_node.inputs[0])

file_output_node = scene_nodes.new('CompositorNodeOutputFile')
file_output_node.location = Vector((300, -150))

output_base_path = '/home/prafulls/outputs/' + str(argv[1]).zfill(6)
if not os.path.exists(output_base_path):
    os.mkdir(output_base_path)
file_output_node.base_path = output_base_path
file_output_node.format.compression = 0
file_output_node.format.color_mode = 'RGB'
scene_node_tree.links.new(render_layers_node.outputs[0], file_output_node.inputs[0])

file_output_node.file_slots.new(name='depth')
file_output_node.file_slots['depth'].use_node_format = False
file_output_node.file_slots['depth'].format.file_format = 'OPEN_EXR'
file_output_node.file_slots['depth'].format.color_depth = '16'
scene_node_tree.links.new(render_layers_node.outputs[2], file_output_node.inputs[1])

file_output_node.file_slots.new(name='segmentation')
file_output_node.file_slots['segmentation'].use_node_format = False
file_output_node.file_slots['segmentation'].format.file_format = 'OPEN_EXR'
file_output_node.file_slots['segmentation'].format.color_depth = '16'
scene_node_tree.links.new(render_layers_node.outputs[4], file_output_node.inputs[2])

file_output_node.file_slots.new(name='normal')
file_output_node.file_slots['normal'].use_node_format = False
file_output_node.file_slots['normal'].format.file_format = 'OPEN_EXR'
file_output_node.file_slots['normal'].format.color_depth = '16'
scene_node_tree.links.new(render_layers_node.outputs[3], file_output_node.inputs[3])

# Add a camera
def get_all_bboxes():
    names = []
    bboxes = []
    for obj in bpy.data.objects:
        if "lamp" in obj.name:
            continue
        print(obj.name)
        bbox = [v[:] for v in obj.bound_box]
        names.append(obj.name)
        bboxes.append(bbox)
    return names, bboxes

def sample_camera_location():
    location = get_random_camera_location()
    while not acceptable_camera_location(location, dist_thresh):
        location = get_random_camera_location()
    return location

def find_distant_object(location):
    max_dist = -1
    look_at_loc = None
    locs = []
    for obj in bpy.data.objects:
        if 'object' not in obj.name:
            continue
        dist_obj = np.linalg.norm(location - obj.location)
        locs.append(np.array(obj.location))
        if dist_obj > max_dist:
            max_dist = dist_obj
            look_at_loc = obj.location
    look_at_loc = np.median(locs, 0)
    return look_at_loc


names, bboxes = get_all_bboxes()
position = sample_location(-floor_length/2, floor_length/2, dist_thresh)
target = find_distant_object(position)
up = np.array([0, 0, 1])

matrix_world_camera = Matrix(makeLookAt(position, target, up))
location, rotation, scale = matrix_world_camera.decompose()

bpy.ops.object.camera_add(enter_editmode=False,
                          align='VIEW',
                          location=location,
                          rotation=rotation.to_euler(),
                          scale=(1, 1, 1))

cam_obj = bpy.data.objects["Camera"]
cam = bpy.data.cameras["Camera"]
cam_obj.data.lens = 24
cam.sensor_fit = "VERTICAL"
bpy.context.scene.camera = bpy.data.objects["Camera"]
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 1024


bpy.context.scene.render.filepath = os.path.join(output_base_path, "frame_")
bpy.ops.render.render(write_still=True)