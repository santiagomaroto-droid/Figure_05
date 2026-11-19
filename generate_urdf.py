"""
Figure 05 URDF Generator (v13.0.0)
Generates a complete, standard 32-link URDF for Figure 05 humanoid robot with
exact 54.50 kg total dry mass and analytical rigid-body inertia tensors for
PyBullet, MuJoCo, Pinocchio, and Isaac Sim / Lab.
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def box_inertia(m: float, dx: float, dy: float, dz: float):
    """Exact principal moments of inertia for a uniform solid cuboid."""
    return [
        round((m / 12.0) * (dy**2 + dz**2), 6),
        round((m / 12.0) * (dx**2 + dz**2), 6),
        round((m / 12.0) * (dx**2 + dy**2), 6),
    ]


def cyl_inertia(m: float, r: float, h: float):
    """Exact principal moments of inertia for a Z-aligned solid cylinder."""
    ixx_iyy = round((m / 12.0) * (3.0 * r**2 + h**2), 6)
    izz = round(0.5 * m * r**2, 6)
    return [ixx_iyy, ixx_iyy, izz]


def sphere_inertia(m: float, r: float):
    """Exact principal moments of inertia for a uniform solid sphere."""
    val = round(0.4 * m * r**2, 6)
    return [val, val, val]


def shape_inertia(m: float, geom):
    if geom[0] == "box":
        return box_inertia(m, geom[1], geom[2], geom[3])
    if geom[0] == "cylinder":
        return cyl_inertia(m, geom[1], geom[2])
    return sphere_inertia(m, geom[1])


def create_figure05_urdf():
    robot = ET.Element("robot", name="figure_05")

    materials = {
        "CarbonBlack": "0.08 0.08 0.09 1.0",
        "AnodizedCyan": "0.02 0.52 0.78 1.0",
        "PolishedTitanium": "0.65 0.68 0.72 1.0",
        "SensorGlass": "0.15 0.20 0.25 0.9",
        "SyntheticSkin": "0.12 0.14 0.18 1.0",
    }

    for name, rgba in materials.items():
        mat = ET.SubElement(robot, "material", name=name)
        ET.SubElement(mat, "color", rgba=rgba)

    def add_link(parent_elem, link_name, mass, visual_geom, col_geom, mat_name, origin_xyz="0 0 0"):
        link = ET.SubElement(parent_elem, "link", name=link_name)
        inertia = shape_inertia(mass, visual_geom)

        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=origin_xyz, rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(mass))
        ET.SubElement(
            inertial,
            "inertia",
            ixx=str(inertia[0]),
            ixy="0.0",
            ixz="0.0",
            iyy=str(inertia[1]),
            iyz="0.0",
            izz=str(inertia[2]),
        )

        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz=origin_xyz, rpy="0 0 0")
        vis_geom = ET.SubElement(visual, "geometry")
        if visual_geom[0] == "box":
            ET.SubElement(vis_geom, "box", size=f"{visual_geom[1]} {visual_geom[2]} {visual_geom[3]}")
        elif visual_geom[0] == "cylinder":
            ET.SubElement(vis_geom, "cylinder", radius=str(visual_geom[1]), length=str(visual_geom[2]))
        elif visual_geom[0] == "sphere":
            ET.SubElement(vis_geom, "sphere", radius=str(visual_geom[1]))
        ET.SubElement(visual, "material", name=mat_name)

        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=origin_xyz, rpy="0 0 0")
        c_geom = ET.SubElement(collision, "geometry")
        if col_geom[0] == "box":
            ET.SubElement(c_geom, "box", size=f"{col_geom[1]} {col_geom[2]} {col_geom[3]}")
        elif col_geom[0] == "cylinder":
            ET.SubElement(c_geom, "cylinder", radius=str(col_geom[1]), length=str(col_geom[2]))
        elif col_geom[0] == "sphere":
            ET.SubElement(c_geom, "sphere", radius=str(col_geom[1]))

        return link

    def add_joint(parent_elem, joint_name, j_type, parent_link, child_link, origin_xyz, origin_rpy, axis, effort, velocity, lower, upper):
        joint = ET.SubElement(parent_elem, "joint", name=joint_name, type=j_type)
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=child_link)
        ET.SubElement(joint, "origin", xyz=origin_xyz, rpy=origin_rpy)
        if j_type != "fixed":
            ET.SubElement(joint, "axis", xyz=axis)
            ET.SubElement(joint, "limit", lower=str(lower), upper=str(upper), effort=str(effort), velocity=str(velocity))
            ET.SubElement(joint, "dynamics", damping="0.1", friction="0.05")
        return joint

    # 1. Base / Pelvis (Structural Monocoque Battery: 8.80 kg, Total System Mass = 54.50 kg)
    add_link(robot, "pelvis", 8.80, ["box", 0.24, 0.26, 0.16], ["box", 0.24, 0.26, 0.16], "CarbonBlack")

    # 2. Torso / Spine (3 DoF -> 12.40 kg)
    add_link(robot, "torso_yaw_link", 1.20, ["cylinder", 0.052, 0.07], ["cylinder", 0.052, 0.07], "AnodizedCyan")
    add_joint(robot, "torso_yaw_joint", "revolute", "pelvis", "torso_yaw_link", "0 0 0.10", "0 0 0", "0 0 1", 220, 4.0, -1.57, 1.57)

    add_link(robot, "torso_pitch_link", 1.40, ["cylinder", 0.052, 0.07], ["cylinder", 0.052, 0.07], "PolishedTitanium")
    add_joint(robot, "torso_pitch_joint", "revolute", "torso_yaw_link", "torso_pitch_link", "0 0 0.08", "0 0 0", "0 1 0", 350, 3.5, -0.6, 1.2)

    add_link(robot, "upper_chest", 9.80, ["box", 0.28, 0.31, 0.36], ["box", 0.28, 0.31, 0.36], "CarbonBlack", origin_xyz="0 0 0.18")
    add_joint(robot, "torso_roll_joint", "revolute", "torso_pitch_link", "upper_chest", "0 0 0.06", "0 0 0", "1 0 0", 220, 4.0, -0.5, 0.5)

    # 3. Head & Vision Suite (1.40 kg)
    add_link(robot, "neck_yaw_link", 0.35, ["cylinder", 0.038, 0.05], ["cylinder", 0.038, 0.05], "AnodizedCyan")
    add_joint(robot, "neck_yaw_joint", "revolute", "upper_chest", "neck_yaw_link", "0 0 0.38", "0 0 0", "0 0 1", 35, 6.0, -1.8, 1.8)

    add_link(robot, "head_pitch_link", 1.05, ["sphere", 0.095], ["sphere", 0.095], "SensorGlass")
    add_joint(robot, "head_pitch_joint", "revolute", "neck_yaw_link", "head_pitch_link", "0 0 0.06", "0 0 0", "0 1 0", 35, 6.0, -0.8, 0.9)

    # 4. Arms (5.15 kg per arm = 10.30 kg)
    for side, sign in [("left", 1.0), ("right", -1.0)]:
        add_link(robot, f"{side}_shoulder_base", 0.90, ["cylinder", 0.048, 0.07], ["cylinder", 0.048, 0.07], "AnodizedCyan")
        add_joint(robot, f"{side}_shoulder_pitch_joint", "revolute", "upper_chest", f"{side}_shoulder_base", f"0 {sign * 0.21} 0.28", "0 0 0", "0 1 0", 220, 5.0, -3.14, 2.0)

        add_link(robot, f"{side}_shoulder_roll_link", 0.70, ["cylinder", 0.042, 0.07], ["cylinder", 0.042, 0.07], "PolishedTitanium")
        add_joint(robot, f"{side}_shoulder_roll_joint", "revolute", f"{side}_shoulder_base", f"{side}_shoulder_roll_link", f"0 {sign * 0.04} 0", "0 0 0", "1 0 0", 220, 5.0, -0.4 if sign > 0 else -2.0, 2.0 if sign > 0 else 0.4)

        add_link(robot, f"{side}_upper_arm", 1.30, ["cylinder", 0.036, 0.26], ["cylinder", 0.036, 0.26], "CarbonBlack", origin_xyz="0 0 -0.13")
        add_joint(robot, f"{side}_shoulder_yaw_joint", "revolute", f"{side}_shoulder_roll_link", f"{side}_upper_arm", f"0 {sign * 0.03} 0", "0 0 0", "0 0 1", 140, 6.0, -2.5, 2.5)

        add_link(robot, f"{side}_elbow_link", 0.50, ["cylinder", 0.034, 0.06], ["cylinder", 0.034, 0.06], "AnodizedCyan")
        add_joint(robot, f"{side}_elbow_joint", "revolute", f"{side}_upper_arm", f"{side}_elbow_link", "0 0 -0.26", "0 0 0", "0 1 0", 180, 6.0, -2.6, 0.0)

        add_link(robot, f"{side}_forearm", 0.75, ["cylinder", 0.030, 0.24], ["cylinder", 0.030, 0.24], "CarbonBlack", origin_xyz="0 0 -0.12")
        add_joint(robot, f"{side}_forearm_roll_joint", "revolute", f"{side}_elbow_link", f"{side}_forearm", "0 0 0", "0 0 0", "0 0 1", 75, 8.0, -3.14, 3.14)

        add_link(robot, f"{side}_wrist_pitch_link", 0.25, ["cylinder", 0.028, 0.04], ["cylinder", 0.028, 0.04], "PolishedTitanium")
        add_joint(robot, f"{side}_wrist_pitch_joint", "revolute", f"{side}_forearm", f"{side}_wrist_pitch_link", "0 0 -0.24", "0 0 0", "0 1 0", 55, 8.0, -1.4, 1.4)

        add_link(robot, f"{side}_hand_palm", 0.75, ["box", 0.04, 0.084, 0.11], ["box", 0.04, 0.084, 0.11], "SyntheticSkin", origin_xyz="0 0 -0.055")
        add_joint(robot, f"{side}_wrist_roll_joint", "revolute", f"{side}_wrist_pitch_link", f"{side}_hand_palm", "0 0 0", "0 0 0", "1 0 0", 55, 8.0, -1.2, 1.2)

    # 5. Legs (10.80 kg per leg = 21.60 kg; Total Robot Mass = 54.50 kg)
    for side, sign in [("left", 1.0), ("right", -1.0)]:
        add_link(robot, f"{side}_hip_yaw_link", 1.10, ["cylinder", 0.052, 0.06], ["cylinder", 0.052, 0.06], "AnodizedCyan")
        add_joint(robot, f"{side}_hip_yaw_joint", "revolute", "pelvis", f"{side}_hip_yaw_link", f"0 {sign * 0.105} -0.08", "0 0 0", "0 0 1", 240, 5.0, -1.2, 1.2)

        add_link(robot, f"{side}_hip_roll_link", 1.30, ["cylinder", 0.052, 0.06], ["cylinder", 0.052, 0.06], "PolishedTitanium")
        add_joint(robot, f"{side}_hip_roll_joint", "revolute", f"{side}_hip_yaw_link", f"{side}_hip_roll_link", "0 0 -0.04", "0 0 0", "1 0 0", 350, 4.5, -0.6 if sign > 0 else -0.4, 0.4 if sign > 0 else 0.6)

        add_link(robot, f"{side}_thigh", 3.80, ["cylinder", 0.048, 0.40], ["cylinder", 0.048, 0.40], "CarbonBlack", origin_xyz="0 0 -0.20")
        add_joint(robot, f"{side}_hip_pitch_joint", "revolute", f"{side}_hip_roll_link", f"{side}_thigh", "0 0 0", "0 0 0", "0 1 0", 510, 4.0, -1.8, 1.2)

        add_link(robot, f"{side}_shank", 2.80, ["cylinder", 0.042, 0.40], ["cylinder", 0.042, 0.40], "CarbonBlack", origin_xyz="0 0 -0.20")
        add_joint(robot, f"{side}_knee_joint", "revolute", f"{side}_thigh", f"{side}_shank", "0 0 -0.40", "0 0 0", "0 1 0", 510, 4.0, 0.0, 2.5)

        add_link(robot, f"{side}_ankle_link", 0.50, ["cylinder", 0.036, 0.05], ["cylinder", 0.036, 0.05], "AnodizedCyan")
        add_joint(robot, f"{side}_ankle_pitch_joint", "revolute", f"{side}_shank", f"{side}_ankle_link", "0 0 -0.40", "0 0 0", "0 1 0", 220, 6.0, -0.9, 0.9)

        add_link(robot, f"{side}_foot", 1.30, ["box", 0.26, 0.136, 0.04], ["box", 0.26, 0.136, 0.04], "PolishedTitanium", origin_xyz="0.035 0 -0.025")
        add_joint(robot, f"{side}_ankle_roll_joint", "revolute", f"{side}_ankle_link", f"{side}_foot", "0 0 0", "0 0 0", "1 0 0", 180, 6.0, -0.5, 0.5)

    raw_xml = ET.tostring(robot, encoding="utf-8")
    reparsed = minidom.parseString(raw_xml)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    out_path = os.path.join(os.path.dirname(__file__), "figure05_kinematics.urdf")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    print(f"[+] Successfully generated analytic-inertia Figure 05 URDF at: {out_path}")


if __name__ == "__main__":
    create_figure05_urdf()
