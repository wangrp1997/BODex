#!/usr/bin/env python3
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import math

import trimesh
import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def _parse_origin_transform(origin_elem):
    """Parse URDF <origin xyz='...' rpy='...'> into 4x4 transform."""
    T = trimesh.transformations.identity_matrix()
    if origin_elem is None:
        return T

    xyz = origin_elem.attrib.get("xyz", "0 0 0")
    rpy = origin_elem.attrib.get("rpy", "0 0 0")
    tx, ty, tz = [float(v) for v in xyz.split()]
    rr, rp, ry = [float(v) for v in rpy.split()]

    # URDF uses fixed-axis RPY extrinsic rotations about X,Y,Z.
    R = trimesh.transformations.euler_matrix(rr, rp, ry, axes="sxyz")
    Tt = trimesh.transformations.translation_matrix([tx, ty, tz])
    return trimesh.transformations.concatenate_matrices(Tt, R)


def resolve_package_path(filename: str, asset_root: Path):
    if filename.startswith("package://"):
        # package://dexhand_description/meshes/xxx.STL -> meshes/xxx.STL
        idx = filename.find("/meshes/")
        if idx != -1:
            return asset_root / filename[idx + 1 :]
        return None
    return asset_root / filename


def build_link_mesh_map(urdf_path: Path, asset_root: Path):
    link_mesh = {}
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        link_name = link.attrib.get("name")
        # Prefer collision mesh; fallback to visual mesh.
        for section in ("collision", "visual"):
            sec = link.find(section)
            if sec is None:
                continue
            geom = sec.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is None:
                continue
            fn = mesh.attrib.get("filename")
            if not fn:
                continue
            resolved = resolve_package_path(fn, asset_root)
            if resolved is not None:
                origin = sec.find("origin")
                tf = _parse_origin_transform(origin)
                link_mesh[link_name] = {"path": resolved, "tf": tf}
                break
    return link_mesh


def build_urdf_graph(urdf_path: Path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    all_links = [l.attrib["name"] for l in root.findall("link")]
    child_links = set()
    children_map = {lnk: [] for lnk in all_links}
    for j in root.findall("joint"):
        p = j.find("parent")
        c = j.find("child")
        if p is None or c is None:
            continue
        parent = p.attrib["link"]
        child = c.attrib["link"]
        child_links.add(child)
        axis_elem = j.find("axis")
        axis = [1.0, 0.0, 0.0]
        if axis_elem is not None and "xyz" in axis_elem.attrib:
            axis = [float(v) for v in axis_elem.attrib["xyz"].split()]
        children_map.setdefault(parent, []).append(
            {
                "name": j.attrib.get("name", ""),
                "type": j.attrib.get("type", "fixed"),
                "parent": parent,
                "child": child,
                "origin": _parse_origin_transform(j.find("origin")),
                "axis": axis,
            }
        )
    root_links = [lnk for lnk in all_links if lnk not in child_links]
    return children_map, root_links


def _joint_motion_tf(joint_type, axis, q):
    if joint_type in ("revolute", "continuous"):
        angle = float(q)
        n = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
        if n < 1e-12:
            return trimesh.transformations.identity_matrix()
        ax = [axis[0] / n, axis[1] / n, axis[2] / n]
        return trimesh.transformations.rotation_matrix(angle, ax)
    if joint_type == "prismatic":
        return trimesh.transformations.translation_matrix(
            [float(q) * axis[0], float(q) * axis[1], float(q) * axis[2]]
        )
    return trimesh.transformations.identity_matrix()


def compute_link_world_tf(children_map, root_links, joint_positions=None):
    if joint_positions is None:
        joint_positions = {}
    world_tf = {}

    def dfs(link_name, T_world):
        world_tf[link_name] = T_world
        for j in children_map.get(link_name, []):
            q = float(joint_positions.get(j["name"], 0.0))
            T = trimesh.transformations.concatenate_matrices(
                T_world, j["origin"], _joint_motion_tf(j["type"], j["axis"], q)
            )
            dfs(j["child"], T)

    for root in root_links:
        dfs(root, trimesh.transformations.identity_matrix())
    return world_tf


def make_scene(mesh_root: Path, link: str, spheres, link_mesh_map=None):
    objs = []
    mesh_path = None
    mesh_tf = None
    if link_mesh_map is not None and link in link_mesh_map:
        mesh_path = link_mesh_map[link]["path"]
        mesh_tf = link_mesh_map[link]["tf"]
    else:
        mesh_path = mesh_root / f"{link}.STL"
    if mesh_path.exists():
        mesh = trimesh.load_mesh(mesh_path)
        if mesh_tf is not None:
            mesh.apply_transform(mesh_tf)
        mesh.visual.face_colors = [180, 180, 180, 120]
        objs.append(mesh)

    for i, s in enumerate(spheres):
        c = s["center"]
        r = float(s["radius"])
        sph = trimesh.creation.icosphere(subdivisions=2, radius=r)
        sph.apply_translation(c)
        # Highlight first sphere a bit differently for orientation.
        if i == 0:
            sph.visual.face_colors = [0, 128, 255, 120]
        else:
            sph.visual.face_colors = [255, 0, 0, 95]
        objs.append(sph)
    return trimesh.Scene(objs)


def _load_link_mesh(mesh_root: Path, link: str, link_mesh_map=None):
    mesh_path = None
    mesh_tf = None
    if link_mesh_map is not None and link in link_mesh_map:
        mesh_path = link_mesh_map[link]["path"]
        mesh_tf = link_mesh_map[link]["tf"]
    else:
        mesh_path = mesh_root / f"{link}.STL"
    if not mesh_path.exists():
        return None
    mesh = trimesh.load_mesh(mesh_path)
    if mesh_tf is not None:
        mesh.apply_transform(mesh_tf)
    return mesh


def make_scene_all(mesh_root: Path, collision_spheres: dict, show_spheres=True, link_mesh_map=None):
    objs = []
    for link, spheres in collision_spheres.items():
        mesh = _load_link_mesh(mesh_root, link, link_mesh_map=link_mesh_map)
        if mesh is not None:
            mesh.visual.face_colors = [180, 180, 180, 90]
            objs.append(mesh)
        if show_spheres:
            for s in spheres:
                c = s["center"]
                r = float(s["radius"])
                sph = trimesh.creation.icosphere(subdivisions=2, radius=r)
                sph.apply_translation(c)
                sph.visual.face_colors = [255, 0, 0, 70]
                objs.append(sph)
    return trimesh.Scene(objs)


def make_scene_all_urdf(
    mesh_root: Path,
    collision_spheres: dict,
    link_mesh_map,
    link_world_tf,
    show_spheres=True,
):
    objs = []
    for link, spheres in collision_spheres.items():
        mesh = _load_link_mesh(mesh_root, link, link_mesh_map=link_mesh_map)
        if mesh is not None:
            T_world = link_world_tf.get(link, trimesh.transformations.identity_matrix())
            mesh.apply_transform(T_world)
            mesh.visual.face_colors = [180, 180, 180, 90]
            objs.append(mesh)
        if show_spheres:
            T_world = link_world_tf.get(link, trimesh.transformations.identity_matrix())
            for s in spheres:
                c = s["center"]
                r = float(s["radius"])
                sph = trimesh.creation.icosphere(subdivisions=2, radius=r)
                sph.apply_translation(c)
                sph.apply_transform(T_world)
                sph.visual.face_colors = [255, 0, 0, 70]
                objs.append(sph)
    return trimesh.Scene(objs)


def print_link_list(links, current_idx):
    print("\n==== Links ====")
    for i, lk in enumerate(links):
        marker = "->" if i == current_idx else "  "
        print(f"{marker} [{i:02d}] {lk}")
    print("================")


def print_current_spheres(link, spheres):
    print(f"\nCurrent link: {link}")
    if not spheres:
        print("  (no spheres)")
        return
    for i, s in enumerate(spheres):
        c = s["center"]
        r = s["radius"]
        print(f"  [{i}] center={c}, radius={r}")


def parse_float_list(text, expected_len):
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != expected_len:
        raise ValueError(f"Expect {expected_len} numbers, got {len(vals)}")
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Tune collision_spheres with quick mesh+sphere preview."
    )
    parser.add_argument(
        "--spheres",
        default="/home/rw/Documents/BODex/src/curobo/content/configs/robot/spheres/right_botyard_hand.yml",
        help="Path to collision spheres yaml",
    )
    parser.add_argument(
        "--mesh-root",
        default="/home/rw/Documents/BODex/src/curobo/content/assets/robot/botyard_description/meshes",
        help="Directory containing per-link STL files",
    )
    parser.add_argument(
        "--urdf",
        default="",
        help="Optional URDF path for generic link->mesh mapping",
    )
    parser.add_argument(
        "--asset-root",
        default="",
        help="Optional asset root for URDF mesh filenames (e.g. .../assets/robot/shadow_hand)",
    )
    args = parser.parse_args()

    spheres_path = Path(args.spheres)
    mesh_root = Path(args.mesh_root)
    link_mesh_map = None
    children_map = None
    root_links = None
    link_world_tf = None
    if args.urdf and args.asset_root:
        urdf_path = Path(args.urdf)
        asset_root = Path(args.asset_root)
        link_mesh_map = build_link_mesh_map(urdf_path, asset_root)
        children_map, root_links = build_urdf_graph(urdf_path)
        link_world_tf = compute_link_world_tf(children_map, root_links, joint_positions=None)
        print(f"Loaded URDF link->mesh map: {len(link_mesh_map)} links")

    data = load_yaml(spheres_path)
    if "collision_spheres" not in data:
        raise RuntimeError("YAML missing key: collision_spheres")

    collision_spheres = data["collision_spheres"]
    links = sorted(collision_spheres.keys())
    idx = 0

    print("Controls:")
    print("  [number]    switch to link index")
    print("  n / p       next / previous link")
    print("  v           visualize current link")
    print("  U           visualize all links with URDF FK (mesh+spheres)")
    print("  J           visualize all links with URDF FK (mesh only)")
    print("  e           edit one sphere in current link")
    print("  a           add sphere to current link")
    print("  d           delete sphere from current link")
    print("  s           save yaml")
    print("  q           quit")

    while True:
        print_link_list(links, idx)
        link = links[idx]
        spheres = collision_spheres.get(link, [])
        print_current_spheres(link, spheres)
        raw = input(
            "\ncmd (Enter=visualize current, number/n/p/v/e/a/d/s/q): "
        ).strip()

        if raw == "":
            raw = "v"

        if raw.isdigit():
            new_idx = int(raw)
            if 0 <= new_idx < len(links):
                idx = new_idx
            else:
                print("Index out of range.")
            continue

        if raw == "n":
            idx = (idx + 1) % len(links)
            continue
        if raw == "p":
            idx = (idx - 1) % len(links)
            continue
        if raw == "v":
            scene = make_scene(mesh_root, link, spheres, link_mesh_map=link_mesh_map)
            scene.show()
            continue
        if raw == "U":
            if link_mesh_map is None or link_world_tf is None:
                print("U needs --urdf and --asset-root.")
                continue
            scene = make_scene_all_urdf(
                mesh_root,
                collision_spheres,
                link_mesh_map=link_mesh_map,
                link_world_tf=link_world_tf,
                show_spheres=True,
            )
            scene.show()
            continue
        if raw == "J":
            if link_mesh_map is None or link_world_tf is None:
                print("J needs --urdf and --asset-root.")
                continue
            scene = make_scene_all_urdf(
                mesh_root,
                collision_spheres,
                link_mesh_map=link_mesh_map,
                link_world_tf=link_world_tf,
                show_spheres=False,
            )
            scene.show()
            continue
        if raw == "e":
            if not spheres:
                print("No spheres to edit.")
                continue
            s_idx = input("sphere idx: ").strip()
            if not s_idx.isdigit() or int(s_idx) >= len(spheres):
                print("Invalid sphere idx.")
                continue
            s_idx = int(s_idx)
            cur = spheres[s_idx]
            print(f"Current center={cur['center']}, radius={cur['radius']}")
            c_in = input("new center x,y,z (Enter skip): ").strip()
            r_in = input("new radius (Enter skip): ").strip()
            try:
                if c_in:
                    cur["center"] = parse_float_list(c_in, 3)
                if r_in:
                    cur["radius"] = float(r_in)
                print("Updated.")
            except Exception as e:
                print(f"Update failed: {e}")
            continue
        if raw == "a":
            c_in = input("center x,y,z: ").strip()
            r_in = input("radius: ").strip()
            try:
                c = parse_float_list(c_in, 3)
                r = float(r_in)
                spheres.append({"center": c, "radius": r})
                print("Sphere added.")
            except Exception as e:
                print(f"Add failed: {e}")
            continue
        if raw == "d":
            if not spheres:
                print("No spheres to delete.")
                continue
            s_idx = input("sphere idx to delete: ").strip()
            if not s_idx.isdigit() or int(s_idx) >= len(spheres):
                print("Invalid sphere idx.")
                continue
            del spheres[int(s_idx)]
            print("Deleted.")
            continue
        if raw == "s":
            save_yaml(spheres_path, data)
            print(f"Saved: {spheres_path}")
            continue
        if raw == "q":
            yes = input("Save before quit? (y/N): ").strip().lower()
            if yes == "y":
                save_yaml(spheres_path, data)
                print(f"Saved: {spheres_path}")
            print("Bye.")
            break

        print("Unknown command.")


if __name__ == "__main__":
    main()
