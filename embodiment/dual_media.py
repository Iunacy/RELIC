import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

from .collection import atomic_json
from .dataset import file_digest
from .replay import load_qualified_pair, validate_pair

NAMES = {'openmanipulator_x': 'OpenMANIPULATOR-X / 4 DoF + gripper', 'franka_panda': 'Franka Panda / 7 DoF + gripper'}


def frame_schedule(duration, fps=15, speed=1.):
    if type(fps) is not int or not 1 <= fps <= 30 or type(speed) not in (int, float) or not math.isfinite(speed) or not .25 <= speed <= 8:
        raise ValueError('invalid export frame rate or declared speed')
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid export duration')
    return [min(duration, index*speed/fps) for index in range(math.ceil(duration*fps/speed)+1)]


def compose(images, states, elapsed, speed, telemetry=True):
    from PIL import Image, ImageDraw, ImageFont
    width, height = images[0].shape[1], images[0].shape[0]
    output = Image.new('RGB', (2*width, height+144), '#171b1e')
    draw = ImageDraw.Draw(output)
    font_path = '/usr/share/fonts/gnu-free/FreeSans.otf'
    try:
        font = ImageFont.truetype(font_path, 18)
        small = ImageFont.truetype(font_path, 14)
    except OSError:
        font, small = ImageFont.load_default(size=18), ImageFont.load_default(size=14)
    draw.text((20, 12), 'VIPER / Embodiment comparison', font=font, fill='#eef1f3')
    draw.text((2*width-480, 15), f'PRERECORDED PHYSX STATE REPLAY / {speed:g}x / t = {elapsed:.2f} s', font=small, fill='#b4c2c8')
    for index, (robot, state) in enumerate(states.items()):
        left = index*width
        draw.text((left+20, 44), NAMES[robot], font=font, fill='#eef1f3')
        output.paste(Image.fromarray(images[index]), (left, 72))
        draw.text((left+20, height+83), state['phase'].replace('_', ' ')+' / '+state['controller'].replace('_', ' '), font=small, fill='#c7d9da')
        status = state['status']+(' / final state held' if state['holding_final_state'] else '')
        draw.text((left+20, height+104), status+' / recorded '+f"{state['recorded_time']:.2f} s", font=small, fill='#b4c2c8')
        if telemetry:
            text = 'EE (m) '+', '.join(f'{value:.3f}' for value in state['ee_position'])
            text += (f" / tracking error {state['tracking_error_m']*1000:.2f} mm" if state.get('tracking_error_m') is not None else " / tracking error unavailable")
            draw.text((left+20, height+125), text, font=small, fill='#b4c2c8')
            draw.text((left+20, 77), f"safety {len(state['safety_events'])} / fallback {len(state['fallback_triggers'])}", font=small, fill='#d8e0e5')
    draw.line((width, 40, width, height+144), fill='#353c41')
    return output


def produce(episodes, output, fps=15, speed=1., gif=False, renderer_factory=None):
    import av
    duration = validate_pair(episodes)
    schedule = frame_schedule(duration, fps, speed)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if renderer_factory is None:
        from .replay_render import Renderer
        renderer_factory = Renderer
    renderers, containers, streams = {}, {}, {}
    success = False
    try:
        for robot in episodes:
            renderers[robot] = renderer_factory(robot)
        for variant in ('clean', 'comparison'):
            container = av.open(str(output/(variant+'.mp4.part')), mode='w', format='mp4')
            stream = container.add_stream('libx264', rate=fps)
            stream.width, stream.height = 1280, 624
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '18', 'preset': 'medium'}
            containers[variant], streams[variant] = container, stream
        frame_indices = []
        for index, elapsed in enumerate(schedule):
            states = {robot: episode.state(elapsed) for robot, episode in episodes.items()}
            images = [renderers[robot].render(episode, elapsed) for robot, episode in episodes.items()]
            frame_indices.append({robot: state['frame_index'] for robot, state in states.items()})
            for variant, stream in streams.items():
                image = compose(images, states, elapsed, speed, variant == 'comparison')
                if image.size != (stream.width, stream.height):
                    raise ValueError('renderer dimensions differ from media layout')
                frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = index, Fraction(1, fps)
                for packet in stream.encode(frame):
                    containers[variant].mux(packet)
        for variant, stream in streams.items():
            for packet in stream.encode():
                containers[variant].mux(packet)
            containers[variant].close()
            decoded = 0
            with av.open(str(output/(variant+'.mp4.part'))) as check:
                for frame in check.decode(video=0):
                    if frame.pts is None or abs(float(frame.time)-decoded/fps) > 1e-5:
                        raise ValueError('encoded media timestamps differ from the shared timeline')
                    if decoded in (0, len(schedule)//2, len(schedule)-1):
                        frame.to_image().save(output/f'{variant}-{decoded:04d}.png')
                    decoded += 1
            if decoded != len(schedule):
                raise ValueError('media frame count mismatch')
            (output/(variant+'.mp4.part')).rename(output/(variant+'.mp4'))
        if gif:
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(output/'comparison.mp4'), '-vf',
                'fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3',
                '-loop', '0', str(output/'simulation.gif')], check=True, timeout=120)
            from PIL import Image
            with Image.open(output/'simulation.gif') as check:
                for index in range(check.n_frames):
                    check.seek(index)
                    check.load()
        result = {'schema': 'relic.dual_media.v1', 'mode': 'prerecorded_physical_state_replay',
                  'dynamics_reexecuted': False, 'learned_transfer_demonstrated': False,
                  'simulated_duration': duration, 'playback_speed': speed, 'fps': fps, 'frames': len(schedule),
                  'source_frame_indices': frame_indices,
                  'source_evidence_sha256': {robot: episode.provenance for robot, episode in episodes.items()},
                  'renderers': {robot: renderer.provenance for robot, renderer in renderers.items()},
                  'outputs': {path.name: {'bytes': path.stat().st_size, 'sha256': file_digest(path)}
                              for path in output.iterdir() if path.suffix in ('.mp4', '.gif')}}
        atomic_json(output/'media.json', result)
        success = True
        return result
    finally:
        for container in containers.values():
            container.close()
        for renderer in renderers.values():
            renderer.close()
        if not success:
            atomic_json(output/'failure.json', {'status': 'BLOCKED', 'partial_media_not_validated': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--gif', action='store_true')
    parser.add_argument('--omx-campaign', type=Path)
    parser.add_argument('--panda-campaign', type=Path)
    args = parser.parse_args()
    roots = {key: value for key, value in [('openmanipulator_x', args.omx_campaign), ('franka_panda', args.panda_campaign)] if value}
    result = produce(load_qualified_pair(args.seed, roots), args.output, args.fps, args.speed, args.gif)
    print(json.dumps({key: result[key] for key in ('mode', 'frames', 'outputs')}, indent=2))


if __name__ == '__main__':
    main()
