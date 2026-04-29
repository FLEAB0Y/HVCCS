import argparse
import math
import os
import sys

import bpy


def parse_args():
	argv = sys.argv
	if "--" in argv:
		argv = argv[argv.index("--") + 1 :]
	else:
		argv = argv[1:]

	parser = argparse.ArgumentParser(description="Import an FBX model and render one frame.")
	parser.add_argument(
		"--fbx",
		type=str,
		default="data/rp_carla_rigged_001_zup_t.fbx",
		help="Path to FBX file. Relative path is resolved from project root.",
	)
	parser.add_argument(
		"--out",
		type=str,
		default="res/video_res/render_h36m.png",
		help="Output image path. Relative path is resolved from project root.",
	)
	parser.add_argument("--width", type=int, default=1280, help="Render width.")
	parser.add_argument("--height", type=int, default=720, help="Render height.")
	parser.add_argument("--samples", type=int, default=64, help="Cycles samples.")
	return parser.parse_args(argv)


def project_root_from_script():
	# Script is in tools/, project root is one level up.
	return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path, root):
	if os.path.isabs(path):
		return path
	return os.path.abspath(os.path.join(root, path))


def clear_scene():
	bpy.ops.object.select_all(action="SELECT")
	bpy.ops.object.delete(use_global=False)
	bpy.data.orphans_purge(do_recursive=True)


def import_fbx(filepath):
	if not os.path.exists(filepath):
		raise FileNotFoundError(f"FBX not found: {filepath}")
	bpy.ops.import_scene.fbx(filepath=filepath)


def _look_at_euler(src, dst):
	# Build camera/light Euler so local -Z points to the target.
	dx = dst[0] - src[0]
	dy = dst[1] - src[1]
	dz = dst[2] - src[2]
	dist_xy = math.hypot(dx, dy)

	yaw = math.atan2(dx, dy)
	pitch = math.atan2(dist_xy, dz) - math.pi / 2.0
	roll = 0.0
	return (pitch, 0.0, -yaw + roll)


def scene_bounds():
	mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
	if not mesh_objects:
		return (0.0, 0.0, 0.0), 1.0

	min_x, min_y, min_z = float("inf"), float("inf"), float("inf")
	max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")

	for obj in mesh_objects:
		hx = obj.dimensions.x * 0.5
		hy = obj.dimensions.y * 0.5
		hz = obj.dimensions.z * 0.5
		cx, cy, cz = obj.location.x, obj.location.y, obj.location.z

		min_x = min(min_x, cx - hx)
		min_y = min(min_y, cy - hy)
		min_z = min(min_z, cz - hz)
		max_x = max(max_x, cx + hx)
		max_y = max(max_y, cy + hy)
		max_z = max(max_z, cz + hz)

	center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5)
	size = max(max_x - min_x, max_y - min_y, max_z - min_z)
	radius = max(size * 0.6, 0.8)
	return center, radius


def setup_camera(center, radius):
	cam_data = bpy.data.cameras.new("RenderCamera")
	cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
	bpy.context.scene.collection.objects.link(cam_obj)

	cam_loc = (center[0] + radius * 2.2, center[1] - radius * 2.2, center[2] + radius * 1.3)
	cam_obj.location = cam_loc
	cam_obj.rotation_euler = _look_at_euler(cam_loc, center)

	cam_data.lens = 50
	bpy.context.scene.camera = cam_obj
	return cam_obj


def setup_lights(center, radius):
	key_data = bpy.data.lights.new("KeyLight", type="SUN")
	key_data.energy = 3.0
	key_obj = bpy.data.objects.new("KeyLight", key_data)
	bpy.context.scene.collection.objects.link(key_obj)
	key_obj.location = (center[0] + radius * 2.0, center[1] - radius * 1.6, center[2] + radius * 3.0)
	key_obj.rotation_euler = (math.radians(45.0), math.radians(0.0), math.radians(35.0))

	fill_data = bpy.data.lights.new("FillLight", type="AREA")
	fill_data.energy = 500
	fill_data.size = radius * 2.0
	fill_obj = bpy.data.objects.new("FillLight", fill_data)
	bpy.context.scene.collection.objects.link(fill_obj)
	fill_loc = (center[0] - radius * 1.5, center[1] + radius * 1.2, center[2] + radius * 1.4)
	fill_obj.location = fill_loc
	fill_obj.rotation_euler = _look_at_euler(fill_loc, center)


def setup_render(out_path, width, height, samples):
	scene = bpy.context.scene
	scene.render.engine = "CYCLES"
	scene.cycles.device = "CPU"
	scene.cycles.samples = max(samples, 1)

	scene.render.resolution_x = width
	scene.render.resolution_y = height
	scene.render.resolution_percentage = 100
	scene.render.image_settings.file_format = "PNG"
	scene.render.filepath = out_path

	world = bpy.data.worlds.get("World")
	if world is None:
		world = bpy.data.worlds.new("World")
	scene.world = world
	world.use_nodes = True
	bg = world.node_tree.nodes.get("Background")
	if bg is not None:
		bg.inputs[1].default_value = 0.9


def main():
	args = parse_args()
	root = project_root_from_script()

	fbx_path = resolve_path(args.fbx, root)
	out_path = resolve_path(args.out, root)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)

	clear_scene()
	import_fbx(fbx_path)

	center, radius = scene_bounds()
	setup_camera(center, radius)
	setup_lights(center, radius)
	setup_render(out_path, args.width, args.height, args.samples)

	bpy.ops.render.render(write_still=True)
	print(f"Rendered image saved to: {out_path}")


if __name__ == "__main__":
	main()
