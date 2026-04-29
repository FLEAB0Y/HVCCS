# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
	import bpy as _bpy
	from mathutils import Vector as _Vector
except ImportError as exc:  # pragma: no cover - this script is intended for Blender runtime.
	raise RuntimeError(
		"This script must run inside Blender where bpy and mathutils are available."
	) from exc

bpy: Any = _bpy
Vector = _Vector


H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

# Primary child used to estimate each mapped bone orientation from H36M points.
H36M_PRIMARY_CHILD = {
	0: 7,
	1: 2,
	2: 3,
	4: 5,
	5: 6,
	7: 8,
	8: 9,
	9: 10,
	11: 12,
	12: 13,
	14: 15,
	15: 16,
}

# Explicit map for rp_carla_rigged_001 (carla_zup_t.blend).
H36M_EXPLICIT_MAP_RP_CARLA = {
	0: "hip",
	1: "upperleg_r",
	2: "lowerleg_r",
	3: "foot_r",
	4: "upperleg_l",
	5: "lowerleg_l",
	6: "foot_l",
	7: "spine_02",
	8: "spine_03",
	9: "neck",
	10: "head",
	11: "shoulder_l",
	12: "lowerarm_l",
	13: "hand_l",
	14: "shoulder_r",
	15: "lowerarm_r",
	16: "hand_r",
}

DEFAULT_NPY_PATH = (
	"/Users/twz/demo_sys_user/h36m_pose_cam_1_downsample/test/"
	"S2_cam_1_30fps/Running_37_cam_1_h36m_30fps.npy"
)


@dataclass
class RenderConfig:
	rig_json_path: str
	rig_name: str
	npy_path: str
	output_dir: str
	target_object_name: str
	blend_path: Optional[str] = None
	fps: int = 30
	width: int = 1280
	height: int = 720
	samples: int = 16
	start_frame: int = 1
	max_frames: int = 0
	motion_scale: float = 1.0
	keep_input_axis: bool = True
	use_explicit_map: bool = True
	allow_fallback_when_explicit_missing: bool = False


def project_root_from_script() -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path: str, project_root: str) -> str:
	if os.path.isabs(path):
		return path
	return os.path.abspath(os.path.join(project_root, path))


def to_tjc_from_npy(pose: np.ndarray, coord_dims: int = 3) -> np.ndarray:
	arr = np.asarray(pose)
	arr = np.squeeze(arr)

	if arr.ndim == 3:
		if arr.shape[-1] >= coord_dims:
			return arr[..., :coord_dims]
		if arr.shape[1] >= coord_dims:
			return np.transpose(arr, (0, 2, 1))[..., :coord_dims]
		raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")

	if arr.ndim == 2:
		if arr.shape[1] % coord_dims == 0:
			joints = arr.shape[1] // coord_dims
			return arr.reshape(arr.shape[0], joints, coord_dims)
		if arr.shape[0] % coord_dims == 0:
			joints = arr.shape[0] // coord_dims
			return arr.T.reshape(arr.shape[1], joints, coord_dims)
		raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")

	if arr.ndim == 1:
		if arr.shape[0] % coord_dims != 0:
			raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")
		joints = arr.shape[0] // coord_dims
		return arr.reshape(1, joints, coord_dims)

	raise ValueError(f"unsupported npy pose shape: {arr.shape}")


def load_pose_frames(npy_path: str, num_keypoints: int = 17) -> np.ndarray:
	pose = np.load(npy_path)
	pose_tjc = to_tjc_from_npy(pose, coord_dims=3).astype(np.float32, copy=False)
	if pose_tjc.shape[1] != num_keypoints:
		raise ValueError(
			f"joint count mismatch: expected {num_keypoints}, got {pose_tjc.shape[1]}"
		)
	return pose_tjc


def load_h36m_rig_map(rig_json_path: str, rig_name: str) -> Dict[int, List[str]]:
	with open(rig_json_path, "r", encoding="utf-8") as f:
		obj = json.load(f)

	if rig_name not in obj:
		raise KeyError(f"rig section not found in rig json: {rig_name}")

	rig_obj = obj[rig_name]
	out: Dict[int, List[str]] = {}
	for key, value in rig_obj.items():
		out[int(key)] = [str(x) for x in value]
	return out


def normalize_name(name: str) -> str:
	return "".join(ch for ch in name.lower() if ch.isalnum())


def list_scene_objects() -> None:
	print("Scene objects:")
	for obj in bpy.data.objects:
		parent_name = obj.parent.name if obj.parent else "None"
		print(f"  - {obj.name} (type={obj.type}, parent={parent_name})")


def find_target_object(target_name: str) -> Optional[Any]:
	aliases = [
		target_name,
		"rp_carla_riggrd_001",
		"rp_carla_rigged_001",
	]
	aliases_norm = [normalize_name(x) for x in aliases]

	for obj in bpy.data.objects:
		obj_norm = normalize_name(obj.name)
		if obj_norm in aliases_norm:
			return obj

	for obj in bpy.data.objects:
		obj_norm = normalize_name(obj.name)
		if any(alias in obj_norm for alias in aliases_norm):
			return obj

	return None


def find_armature_for_object(obj: Any) -> Optional[Any]:
	if obj.type == "ARMATURE":
		return obj

	if obj.parent and obj.parent.type == "ARMATURE":
		return obj.parent

	for mod in obj.modifiers:
		if mod.type == "ARMATURE" and mod.object and mod.object.type == "ARMATURE":
			return mod.object

	for child in obj.children:
		if child.type == "ARMATURE":
			return child

	armatures = [x for x in bpy.data.objects if x.type == "ARMATURE"]
	return armatures[0] if armatures else None


def list_armature_bones(armature_obj: Any) -> List[str]:
	bone_names = [pb.name for pb in armature_obj.pose.bones]
	print(f"Armature bones ({len(bone_names)}):")
	for i, name in enumerate(bone_names):
		print(f"  [{i:03d}] {name}")
	return bone_names


def print_joint_mapping_table(joint_bone_map: Dict[int, str], title: str = "H36M joint mapping") -> None:
	print(f"{title}:")
	print("  idx | h36m_name  -> armature_bone")
	print("  ----+-----------------------------")
	for idx in range(17):
		mapped = joint_bone_map.get(idx, "<unmapped>")
		print(f"  {idx:>3} | H36M[{idx:>2}] -> {mapped}")


def _best_bone_match(
	pose_bones: List[Any],
	candidate_labels: List[str],
	fallback_keywords: List[str],
) -> Tuple[Optional[Any], str]:
	normalized_to_bone = {normalize_name(pb.name): pb for pb in pose_bones}

	normalized_candidates = [normalize_name(x) for x in candidate_labels if x]
	for cand in normalized_candidates:
		if cand in normalized_to_bone:
			return normalized_to_bone[cand], "exact_candidate"

	contains_matches = []
	for pb in pose_bones:
		pb_norm = normalize_name(pb.name)
		score = 0
		for cand in normalized_candidates:
			if cand and (cand in pb_norm or pb_norm in cand):
				score += max(len(cand), 1)
		if score > 0:
			contains_matches.append((score, pb))

	if contains_matches:
		contains_matches.sort(key=lambda x: x[0], reverse=True)
		return contains_matches[0][1], "contains_candidate"

	keyword_matches = []
	norm_keywords = [normalize_name(x) for x in fallback_keywords if x]
	for pb in pose_bones:
		pb_norm = normalize_name(pb.name)
		score = 0
		for kw in norm_keywords:
			if kw and kw in pb_norm:
				score += len(kw)
		if score > 0:
			keyword_matches.append((score, pb))

	if keyword_matches:
		keyword_matches.sort(key=lambda x: x[0], reverse=True)
		return keyword_matches[0][1], "fallback_keyword"

	return None, "not_found"


def build_joint_bone_map(
	armature_obj: Any,
	h36m_rig_map: Dict[int, List[str]],
	explicit_map: Optional[Dict[int, str]] = None,
	allow_fallback_when_explicit_missing: bool = False,
) -> Dict[int, str]:
	fallback_map = {
		0: ["pelvis", "hip", "root"],
		1: ["rthigh", "rightthigh", "rightupleg", "rhip"],
		2: ["rcalf", "rightleg", "rknee"],
		3: ["rfoot", "rank", "rightankle"],
		4: ["lthigh", "leftthigh", "leftupleg", "lhip"],
		5: ["lcalf", "leftleg", "lknee"],
		6: ["lfoot", "lank", "leftankle"],
		7: ["spine", "spine1"],
		8: ["spine2", "spine3", "chest", "thorax", "clav"],
		9: ["neck"],
		10: ["head"],
		11: ["lclavicle", "leftshoulder", "lupperarm"],
		12: ["lforearm", "leftelbow", "lelb"],
		13: ["lhand", "leftwrist", "lwrt"],
		14: ["rclavicle", "rightshoulder", "rupperarm"],
		15: ["rforearm", "rightelbow", "relb"],
		16: ["rhand", "rightwrist", "rwrt"],
	}

	pose_bones = list(armature_obj.pose.bones)
	pose_bone_names = {pb.name for pb in pose_bones}
	mapping: Dict[int, str] = {}

	if explicit_map:
		print("Applying explicit H36M->bone map first...")
		for idx in sorted(explicit_map.keys()):
			bone_name = explicit_map[idx]
			if bone_name in pose_bone_names:
				mapping[idx] = bone_name
				print(f"Mapped H36M[{idx}] -> {bone_name} (reason=explicit_map)")
			else:
				print(
					f"Warning: explicit map bone missing for H36M[{idx}] "
					f"(bone={bone_name})"
				)

		if not allow_fallback_when_explicit_missing:
			return mapping

	for idx in sorted(h36m_rig_map.keys()):
		if idx in mapping:
			continue

		candidates = h36m_rig_map.get(idx, [])
		fallback = fallback_map.get(idx, [])
		pb, reason = _best_bone_match(pose_bones, candidates, fallback)
		if pb is not None:
			mapping[idx] = pb.name
			print(
				f"Mapped H36M[{idx}] -> {pb.name} "
				f"(reason={reason}, candidates={candidates}, fallback={fallback})"
			)
		else:
			print(
				f"Warning: no bone matched for H36M[{idx}] "
				f"(reason={reason}, candidates={candidates}, fallback={fallback})"
			)

	return mapping


def get_rest_heads(armature_obj: Any, joint_bone_map: Dict[int, str]) -> Dict[int, Any]:
	out: Dict[int, Any] = {}
	for idx, bone_name in joint_bone_map.items():
		data_bone = armature_obj.data.bones.get(bone_name)
		if data_bone is None:
			continue
		out[idx] = data_bone.head_local.copy()
	return out


def remap_pose_axes(frame_pose: np.ndarray, keep_input_axis: bool) -> np.ndarray:
	if keep_input_axis:
		return frame_pose

	# Default remap for common h36m coordinates -> Blender coordinates.
	out = np.empty_like(frame_pose)
	out[:, 0] = frame_pose[:, 0]
	out[:, 1] = frame_pose[:, 2]
	out[:, 2] = -frame_pose[:, 1]
	return out


def apply_pose_frame(
	armature_obj: Any,
	joint_bone_map: Dict[int, str],
	rest_heads: Dict[int, Any],
	frame_pose: np.ndarray,
	reference_pose: np.ndarray,
	motion_scale: float,
) -> None:
	pose_bones = armature_obj.pose.bones
	data_bones = armature_obj.data.bones

	target_heads: Dict[int, Any] = {}
	for idx in joint_bone_map.keys():
		if idx not in rest_heads:
			continue

		delta = (frame_pose[idx] - reference_pose[idx]) * motion_scale
		target_heads[idx] = rest_heads[idx] + Vector((float(delta[0]), float(delta[1]), float(delta[2])))

	# Reset mapped bones to avoid accumulating transform drift.
	for bone_name in set(joint_bone_map.values()):
		pb = pose_bones.get(bone_name)
		if pb is None:
			continue
		pb.location = Vector((0.0, 0.0, 0.0))
		pb.rotation_mode = "QUATERNION"
		pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

	# Pelvis translation only. Other joints are driven by rotation to reduce stretching.
	root_idx = 0
	root_bone_name = joint_bone_map.get(root_idx)
	if root_bone_name and root_idx in target_heads and root_idx in rest_heads:
		root_pb = pose_bones.get(root_bone_name)
		if root_pb is not None:
			root_pb.location = target_heads[root_idx] - rest_heads[root_idx]

	for parent_idx, child_idx in H36M_PRIMARY_CHILD.items():
		if parent_idx not in joint_bone_map or child_idx not in joint_bone_map:
			continue
		if parent_idx not in target_heads or child_idx not in target_heads:
			continue

		bone_name = joint_bone_map[parent_idx]
		pb = pose_bones.get(bone_name)
		db = data_bones.get(bone_name)
		if pb is None or db is None:
			continue

		rest_dir = db.tail_local - db.head_local
		target_dir = target_heads[child_idx] - target_heads[parent_idx]
		if rest_dir.length < 1e-8 or target_dir.length < 1e-8:
			continue

		q_delta = rest_dir.normalized().rotation_difference(target_dir.normalized())
		pb.rotation_mode = "QUATERNION"
		pb.rotation_quaternion = q_delta


def setup_render(scene: Any, output_dir: str, fps: int, width: int, height: int, samples: int) -> None:
	engine_candidates = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"]
	selected_engine = None
	for eng in engine_candidates:
		try:
			scene.render.engine = eng
			selected_engine = eng
			break
		except Exception:
			continue
	if selected_engine is None:
		raise RuntimeError("No Eevee engine is available in current Blender runtime")
	print(f"Render engine: {selected_engine}")

	scene.render.resolution_x = int(width)
	scene.render.resolution_y = int(height)
	scene.render.resolution_percentage = 100
	scene.render.fps = int(max(1, fps))
	scene.render.image_settings.file_format = "PNG"

	if hasattr(scene, "eevee"):
		scene.eevee.taa_render_samples = int(max(1, samples))

	if scene.camera is None:
		bpy.ops.object.camera_add(location=(0.0, -4.0, 1.7), rotation=(1.4, 0.0, 0.0))
		scene.camera = bpy.context.object

	os.makedirs(output_dir, exist_ok=True)


def render_pose_sequence(
	armature_obj: Any,
	frames_tjc: np.ndarray,
	joint_bone_map: Dict[int, str],
	rest_heads: Dict[int, Any],
	output_dir: str,
	start_frame: int,
	max_frames: int,
	motion_scale: float,
	keep_input_axis: bool,
) -> int:
	if frames_tjc.shape[0] == 0:
		return 0

	total = frames_tjc.shape[0]
	if max_frames > 0:
		total = min(total, max_frames)

	reference_pose = remap_pose_axes(frames_tjc[0], keep_input_axis=keep_input_axis)

	scene = bpy.context.scene
	scene.frame_start = start_frame
	scene.frame_end = start_frame + total - 1

	for i in range(total):
		scene_frame = start_frame + i
		scene.frame_set(scene_frame)

		pose_frame = remap_pose_axes(frames_tjc[i], keep_input_axis=keep_input_axis)
		apply_pose_frame(
			armature_obj=armature_obj,
			joint_bone_map=joint_bone_map,
			rest_heads=rest_heads,
			frame_pose=pose_frame,
			reference_pose=reference_pose,
			motion_scale=motion_scale,
		)

		bpy.context.view_layer.update()

		out_path = os.path.join(output_dir, f"frame_{i:05d}.png")
		scene.render.filepath = out_path
		bpy.ops.render.render(write_still=True)

		if (i + 1) % 10 == 0 or i == 0 or i + 1 == total:
			print(f"Rendered {i + 1}/{total}: {out_path}")

	return total


def main(config: RenderConfig) -> int:
	project_root = project_root_from_script()

	rig_json_path = resolve_path(config.rig_json_path, project_root)
	npy_path = resolve_path(config.npy_path, project_root)
	output_dir = resolve_path(config.output_dir, project_root)
	blend_path = None
	if config.blend_path:
		blend_path = resolve_path(config.blend_path, project_root)

	if blend_path:
		if not os.path.exists(blend_path):
			raise FileNotFoundError(f"blend file not found: {blend_path}")
		bpy.ops.wm.open_mainfile(filepath=blend_path)
		print(f"Opened blend file: {blend_path}")

	if not os.path.exists(rig_json_path):
		raise FileNotFoundError(f"rig json not found: {rig_json_path}")
	if not os.path.exists(npy_path):
		raise FileNotFoundError(f"npy file not found: {npy_path}")

	list_scene_objects()
	target_obj = find_target_object(config.target_object_name)
	if target_obj is None:
		raise RuntimeError(
			f"cannot find target object '{config.target_object_name}' in scene. "
			"Please pass --target-object with an existing object name."
		)
	print(f"Target object: {target_obj.name} (type={target_obj.type})")

	armature_obj = find_armature_for_object(target_obj)
	if armature_obj is None:
		raise RuntimeError(f"cannot find armature for target object: {target_obj.name}")
	print(f"Armature object: {armature_obj.name}")
	list_armature_bones(armature_obj)

	h36m_rig_map = load_h36m_rig_map(rig_json_path, config.rig_name)
	frames_tjc = load_pose_frames(npy_path, num_keypoints=17)
	print(f"Loaded pose frames: {frames_tjc.shape}")

	explicit_map = H36M_EXPLICIT_MAP_RP_CARLA if config.use_explicit_map else None
	joint_bone_map = build_joint_bone_map(
		armature_obj,
		h36m_rig_map,
		explicit_map=explicit_map,
		allow_fallback_when_explicit_missing=config.allow_fallback_when_explicit_missing,
	)
	if not joint_bone_map:
		raise RuntimeError("no H36M joints mapped to pose bones, cannot render")
	if config.use_explicit_map and len(joint_bone_map) < 17 and not config.allow_fallback_when_explicit_missing:
		raise RuntimeError(
			"explicit mapping enabled but incomplete. "
			"Set allow_fallback_when_explicit_missing=True or complete explicit map."
		)
	print_joint_mapping_table(joint_bone_map, title="Effective H36M->armature mapping table")

	rest_heads = get_rest_heads(armature_obj, joint_bone_map)
	if not rest_heads:
		raise RuntimeError("failed to get rest heads from mapped bones")

	setup_render(
		scene=bpy.context.scene,
		output_dir=output_dir,
		fps=config.fps,
		width=config.width,
		height=config.height,
		samples=config.samples,
	)

	rendered = render_pose_sequence(
		armature_obj=armature_obj,
		frames_tjc=frames_tjc,
		joint_bone_map=joint_bone_map,
		rest_heads=rest_heads,
		output_dir=output_dir,
		start_frame=config.start_frame,
		max_frames=config.max_frames,
		motion_scale=float(config.motion_scale),
		keep_input_axis=bool(config.keep_input_axis),
	)

	print(
		"render_done: "
		f"frames={rendered}, "
		f"output_dir={output_dir}, "
		f"rig={config.rig_name}, "
		f"target={target_obj.name}, "
		f"armature={armature_obj.name}"
	)
	return rendered


if __name__ == "__main__":
	cfg = RenderConfig(
		rig_json_path="data/rig.json",
		rig_name="Human3.6M",
		npy_path=DEFAULT_NPY_PATH,
		output_dir="res/video_res/render_h36m_blend",
		target_object_name="rp_carla_riggrd_001",
		blend_path="data/carla_zup_t.blend",
		fps=30,
		width=1280,
		height=720,
		samples=16,
		start_frame=1,
		max_frames=10,
		motion_scale=1.0,
		keep_input_axis=True,
		use_explicit_map=True,
		allow_fallback_when_explicit_missing=False,
	)
	main(cfg)
