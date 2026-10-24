import math
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from .assets import ARM, FINGERS, ROOT, fetch
from .benchmark import BIN, BIN_INNER, BIN_WALL, COLORS, CUBE
from .dataset import DOF, file_digest, read_json
from .panda import ARM as PANDA_ARM, BASE, FINGERS as PANDA_FINGERS, TOOL_OFFSET


def panda_asset(source=None):
    cache = ROOT/'cache/panda_replay'
    manifest = read_json(ROOT/'assets/panda_replay_manifest.json')
    if source is None:
        source = os.environ.get('RELIC_PANDA_DESCRIPTION')
    if source is None:
        import importlib.metadata
        try:
            source = Path(importlib.metadata.distribution('isaacsim').locate_file('isaacsim'))/'exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description'
        except importlib.metadata.PackageNotFoundError:
            source = Path.home()/'.local/share/relic-isaac/venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description'
    source = Path(source)
    for name, digest in manifest['files'].items():
        destination = cache/name
        if not destination.exists():
            original = source/name
            if not original.is_file():
                raise ValueError('Panda replay assets missing; set RELIC_PANDA_DESCRIPTION to the pinned franka_description directory')
            if file_digest(original) != digest:
                raise ValueError('Panda source asset differs from pinned replay manifest')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(original.read_bytes())
        if file_digest(destination) != digest:
            raise ValueError('Panda replay cache checksum mismatch')
    return cache/'robots/panda_arm_hand.urdf'


def panda_visuals(urdf):
    root = urdf.parents[1]
    output = root/'converted_visuals'
    manifest = output/'manifest.json'
    source_hashes = {file.name: file_digest(file) for file in sorted((root/'meshes/visual').glob('*.dae'))}
    if manifest.exists():
        saved = read_json(manifest)
        if saved['source_sha256'] != source_hashes:
            raise ValueError('converted Panda visuals have stale source geometry')
        for part in saved['parts']:
            if file_digest(output/part['file']) != part['sha256']:
                raise ValueError('converted Panda visual cache changed')
        return saved['parts']
    import trimesh
    output.mkdir(exist_ok=True)
    parts = []
    for source in sorted((root/'meshes/visual').glob('*.dae')):
        scene = trimesh.load(source, force='scene')
        for index, node in enumerate(scene.graph.nodes_geometry):
            transform, key = scene.graph[node]
            geometry = scene.geometry[key].copy()
            geometry.apply_transform(transform)
            material = geometry.visual.material
            color = getattr(material, 'baseColorFactor', None)
            rgba = [float(value)/255 for value in color] if color is not None else [.75, .75, .75, 1.]
            filename = f'{source.stem}_{index:03d}.obj'
            geometry.export(output/filename, file_type='obj', include_texture=False)
            parts.append({'body': 'panda_'+source.stem,
                          'source': source.stem, 'file': filename, 'rgba': rgba,
                          'sha256': file_digest(output/filename), 'vertices': len(geometry.vertices)})
    manifest.write_text(json.dumps({'source_sha256': source_hashes, 'parts': parts}, indent=2)+'\n')
    return parts


class Renderer:
    def __init__(self, robot, width=640, height=480):
        import mujoco
        self.mj = mujoco
        if robot not in DOF:
            raise ValueError('unknown replay embodiment')
        self.robot, self.width, self.height = robot, width, height
        urdf = fetch() if robot == 'openmanipulator_x' else panda_asset()
        tree = ET.parse(urdf)
        origins = {}
        for link in tree.findall('link'):
            origin = link.find('visual/origin')
            xyz = [float(value) for value in origin.get('xyz', '0 0 0').split()] if origin is not None else [0., 0., 0.]
            rpy = [float(value) for value in origin.get('rpy', '0 0 0').split()] if origin is not None else [0., 0., 0.]
            roll, pitch, yaw = [value/2 for value in rpy]
            cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
            origins[link.attrib['name']] = (xyz, [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])
        if robot == 'franka_panda':
            for mesh in tree.findall('.//mesh'):
                name = mesh.attrib['filename'].removeprefix('package://franka_description/')
                mesh.attrib['filename'] = str(urdf.parents[1]/name.replace('/visual/', '/collision/').replace('.dae', '.stl'))
        for link in tree.findall('link'):
            for visual in link.findall('visual'):
                link.remove(visual)
        extension = ET.SubElement(tree.getroot(), 'mujoco')
        ET.SubElement(extension, 'compiler', {'fusestatic': 'false', 'discardvisual': 'false'})
        spec = mujoco.MjSpec.from_string(ET.tostring(tree.getroot(), encoding='unicode'))
        visual_parts = []
        if robot == 'franka_panda':
            spec.body('panda_link0').pos = BASE
            visual_parts = panda_visuals(urdf)
            for index, part in enumerate(visual_parts):
                name = f'panda_visual_{index}'
                spec.add_mesh(name=name, file=str(urdf.parents[1]/'converted_visuals'/part['file']))
                spec.add_material(name=name, rgba=part['rgba'], specular=.35, shininess=.3)
                bodies = ['panda_leftfinger', 'panda_rightfinger'] if part['source'] == 'finger' else [part['body']]
                for body_name in bodies:
                    spec.body(body_name).add_geom(name=name+'_'+body_name, type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname=name, material=name, pos=origins[body_name][0], quat=origins[body_name][1], group=2, contype=0, conaffinity=0)
        tool_body = 'link5' if robot == 'openmanipulator_x' else 'panda_hand'
        offset = [.126, 0., 0.] if robot == 'openmanipulator_x' else [0., 0., TOOL_OFFSET]
        spec.body(tool_body).add_site(name='relic_replay_tool', pos=offset, size=[.001, 0, 0], rgba=[0, 0, 0, 0])
        spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
        spec.add_texture(name='studio_sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                         rgb1=[.08, .09, .10], rgb2=[.08, .09, .10], width=256, height=1536)
        spec.worldbody.add_geom(name='replay_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, .1], rgba=[.105, .115, .125, 1])
        spec.worldbody.add_light(pos=[.2, -.2, 1.8], dir=[0, 0, -1], diffuse=[.65, .65, .65],
                                 specular=[.25, .25, .25], bulbradius=.12)
        spec.worldbody.add_light(pos=[-.6, .6, .9], dir=[.4, -.4, -.6], diffuse=[.3, .33, .36], castshadow=False)
        spec.add_material(name='robotis_metal', rgba=[.5, .52, .54, 1], specular=.65, shininess=.55)
        self.cube_bodies = []
        for color, rgba in zip(COLORS, ([.78, .11, .09, 1], [.12, .58, .32, 1], [.09, .35, .8, 1])):
            body = spec.worldbody.add_body(name='recorded_'+color, mocap=True)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE/2]*3, rgba=rgba, contype=0, conaffinity=0)
            self.cube_bodies.append('recorded_'+color)
        x, y, z = BIN
        outer, t = BIN_INNER+.008, .004
        pieces = [('bottom', [outer/2, outer/2, z/2], [x, y, z/2]),
                  ('left', [t/2, outer/2, BIN_WALL/2], [x-(BIN_INNER+t)/2, y, z+BIN_WALL/2]),
                  ('right', [t/2, outer/2, BIN_WALL/2], [x+(BIN_INNER+t)/2, y, z+BIN_WALL/2]),
                  ('front', [BIN_INNER/2, t/2, BIN_WALL/2], [x, y-(BIN_INNER+t)/2, z+BIN_WALL/2]),
                  ('back', [BIN_INNER/2, t/2, BIN_WALL/2], [x, y+(BIN_INNER+t)/2, z+BIN_WALL/2])]
        for name, size, pos in pieces:
            spec.worldbody.add_geom(name='bin_'+name, type=mujoco.mjtGeom.mjGEOM_BOX,
                                    size=size, pos=pos, rgba=[.36, .39, .42, 1], contype=0, conaffinity=0)
        self.model = spec.compile()
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        if robot == 'franka_panda':
            for geom in range(self.model.ngeom):
                if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH and self.model.geom_contype[geom] != 0:
                    self.model.geom_group[geom] = 3
        if robot == 'openmanipulator_x':
            material = self.model.material('robotis_metal').id
            for geom in range(self.model.ngeom):
                if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                    self.model.geom_matid[geom] = material
        self.model.vis.quality.shadowsize = 4096
        self.model.vis.quality.offsamples = 4
        self.model.vis.headlight.ambient[:] = [.3, .3, .3]
        self.model.vis.headlight.diffuse[:] = [.45, .45, .45]
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.joint_names = ARM+FINGERS if robot == 'openmanipulator_x' else PANDA_ARM+PANDA_FINGERS
        self.joints = [self.model.joint(name).id for name in self.joint_names]
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [.13, .035, .10] if robot == 'openmanipulator_x' else [-.04, .035, .32]
        self.camera.distance = .52 if robot == 'openmanipulator_x' else 1.4
        self.camera.azimuth, self.camera.elevation = 130, -26
        self.detail_camera = mujoco.MjvCamera()
        self.detail_camera.lookat[:] = [.16, .04, .04]
        self.detail_camera.distance = .43
        self.detail_camera.azimuth, self.detail_camera.elevation = 130, -48
        self.provenance = {'urdf_sha256': file_digest(urdf), 'renderer': 'MuJoCo observed-state replay',
                           'dynamics_reexecuted': False, 'visual_meshes': 'official detailed Collada visual meshes with source materials' if robot == 'franka_panda' else 'official ROBOTIS STL meshes',
                           'visual_parts': len(visual_parts),
                           'camera': {'lookat': self.camera.lookat.tolist(), 'distance': self.camera.distance,
                                      'azimuth': 130, 'elevation': -26}}

    def render(self, episode, elapsed, trajectory=True):
        import numpy as np
        state = episode.state(elapsed)
        for joint, value, velocity in zip(self.joints, state['joint_position'], state['joint_velocity'], strict=True):
            if not self.model.jnt_range[joint, 0]-.002 <= value <= self.model.jnt_range[joint, 1]+.002:
                raise ValueError('recorded joint exceeds real asset limits')
            self.data.qpos[self.model.jnt_qposadr[joint]] = value
            self.data.qvel[self.model.jnt_dofadr[joint]] = velocity
        for color, name in zip(COLORS, self.cube_bodies, strict=True):
            index = self.model.body_mocapid[self.model.body(name).id]
            pose = state['cube_poses'][color]
            self.data.mocap_pos[index], self.data.mocap_quat[index] = pose[:3], pose[3:]
        self.mj.mj_forward(self.model, self.data)
        error = math.dist(self.data.site_xpos[self.model.site('relic_replay_tool').id], state['ee_position'])
        self.provenance['maximum_fk_error_m'] = max(error, self.provenance.get('maximum_fk_error_m', 0.))
        if error > .0002:
            raise ValueError(f'asset kinematics differ from recorded execution: {error:.6f} m')
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        scene = self.renderer.scene
        if trajectory:
            index = state['frame_index']
            rows = episode.rows[max(0, index-240):index+1:6]
            for previous, current in zip(rows, rows[1:]):
                geom = scene.geoms[scene.ngeom]
                self.mj.mjv_initGeom(geom, self.mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                   np.array([.45, .72, .74, .65], dtype=np.float32))
                self.mj.mjv_connector(geom, self.mj.mjtGeom.mjGEOM_CAPSULE, .0008, previous['ee_position'], current['ee_position'])
                scene.ngeom += 1
            geom = scene.geoms[scene.ngeom]
            self.mj.mjv_initGeom(geom, self.mj.mjtGeom.mjGEOM_SPHERE, [.0025, 0, 0], state['control_target'], np.eye(3).ravel(),
                               np.array([.92, .75, .37, 1], dtype=np.float32))
            scene.ngeom += 1
        result = self.renderer.render().copy()
        if self.robot == 'franka_panda':
            from PIL import Image, ImageDraw, ImageFont
            self.renderer.update_scene(self.data, camera=self.detail_camera, scene_option=self.option)
            detail = Image.fromarray(self.renderer.render()).resize((208, 156), Image.Resampling.LANCZOS)
            image = Image.fromarray(result)
            left, top = 12, self.height-180
            image.paste(detail, (left, top+19))
            draw = ImageDraw.Draw(image)
            draw.rectangle((left, top, left+208, top+19), fill='#242b30')
            draw.text((left+6, top+3), 'Workspace / same recorded time', fill='#e2eaed', font=ImageFont.load_default(size=10))
            draw.rectangle((left, top, left+208, top+175), outline='#4b565e')
            result = np.asarray(image).copy()
        return result

    def close(self):
        self.renderer.close()
