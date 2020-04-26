import bpy
import fnmatch
from bpy import context
import random

print("-----------------------------------------")
print("----------- start the script ------------")
print("-----------------------------------------")
# prepare a scene
frameAnz = 50
scene = bpy.context.scene
scene.frame_start = 0
scene.frame_end = frameAnz
bpy.ops.screen.frame_jump(end=False)
EmptyNames = ["hand_r", "hand_l"]  # list of the emptys that will be used

# find the armument
arm = bpy.data.objects['Game_engine']

bpy.ops.object.mode_set(mode='OBJECT')
print("        1. Armument found")

# Add a temporary empty
for obj in bpy.context.scene.objects:  # chek if the emty from the list is alredy used
    if obj.type == 'EMPTY':
        bpy.data.objects.remove(obj)

print("        2. start adding Emptys")
for emptyName in EmptyNames:
    empty = bpy.ops.object.add()
    bpy.context.scene.objects['Empty'].name = emptyName
    bpy.context.view_layer.objects.active = arm
    print("            add Empty : ", emptyName)
print("        2. all Emptys are added")

for actFrame in range(1, frameAnz + 1):

    bpy.context.scene.frame_set(actFrame)

    # bpy.context.scene.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')  # all obj are reddy go to the pose mode

    for emptyName in EmptyNames:
        bone = arm.pose.bones[emptyName]  # chose the bone that corespond to the empty (same name)
        tail = bone.tail  # bone tail position
        head = bone.head  # bone head position
        empty = bpy.context.scene.objects[emptyName]
        # rotate the point around x_Achsis 90° and divide by 10
        empty.location.x = (head[0]) / 10  # - 0.00002#random.random()/8
        empty.location.y = -(head[2]) / 10 - 0.1  # - 0.00002#random.random()/8
        empty.location.z = (head[1]) / 10  # + 0.00002#random.random()/8
        print("        3. set Empty ", emptyName, " Location")

        # Add IK constraint and update
        constraint = bone.constraints.new('IK')
        constraint.chain_count = 3
        constraint.target = empty
        bone.constraints.update()
        print("        4. move bone to ", emptyName, "Location")

    # add to key frame
    print("        5. add to Key Frame")
    print("----------- end frame", actFrame, " ------------")
    for mesh_obj in bpy.context.scene.objects:
        mesh_obj.keyframe_insert(data_path="location", index=-1)
        print(mesh_obj.name, " added")

# job done: now reset the framekeypointer to 0    
bpy.ops.screen.frame_jump(end=False)
########################################################################
