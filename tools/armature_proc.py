import bpy

def import_fbx(filepath):
    """Import FBX file into Blender."""
    bpy.ops.import_scene.fbx(filepath=filepath)

def get_shapekeys_and_bones_with_dimensions():
    """Retrieve Shape Keys and Armature Bones with their control dimensions."""
    shapekeys = []
    bones_info = []

    # Iterate through objects in the scene
    for obj in bpy.data.objects:
        # Check for Shape Keys
        if obj.type == 'MESH' and obj.data.shape_keys:
            shapekeys.extend([key.name for key in obj.data.shape_keys.key_blocks])

        # Check for Armature Bones
        if obj.type == 'ARMATURE':
            for bone in obj.data.bones:
                # Determine control dimensions
                dimensions = []
                if bone.use_connect:
                    dimensions.append("Connected")
                if bone.use_inherit_rotation:
                    dimensions.append("Inherits Rotation")
                if bone.use_local_location:
                    dimensions.append("Local Location")
                if bone.use_deform:
                    dimensions.append("Deformable")
                
                bones_info.append({
                    "name": bone.name,
                    "dimensions": dimensions
                })

    return shapekeys, bones_info

def main():
    # Path to the FBX file
    fbx_filepath = "/Users/twz/demo_sys_user/HVCCS/data/rp_carla_rigged_001_zup_t.fbx"

    # Import the FBX file
    import_fbx(fbx_filepath)

    # Retrieve Shape Keys and Bones with dimensions
    shapekeys, bones_info = get_shapekeys_and_bones_with_dimensions()

    # Output the results
    print("Shape Keys:")
    for key in shapekeys:
        print(f"  - {key}")

    print("\nBones:")
    for bone in bones_info:
        print(f"  - {bone['name']}")
        print(f"    Dimensions: {', '.join(bone['dimensions']) if bone['dimensions'] else 'None'}")

if __name__ == "__main__":
    main()