import argparse
from fractions import Fraction
from pathlib import Path
from .dataset import load_episode


def render(episode_path, output):
    episode = load_episode(Path(episode_path), expert_only=False)
    if episode.metadata.get('split') != 'evaluation':
        raise ValueError('prerecorded demo requires held-out evaluation footage')
    import av
    from PIL import Image, ImageDraw
    output = Path(output)
    with output.open('xb') as target:
        with av.open(target,mode='w',format='mp4') as container:
            stream = container.add_stream('libx264',rate=30)
            stream.width,stream.height = episode.shape[1],episode.shape[0]
            stream.pix_fmt = 'yuv420p'
            for index,row in enumerate(episode.rows):
                with Image.open(episode.path/row['frame']) as source:
                    image = source.convert('RGB')
                draw = ImageDraw.Draw(image)
                draw.rectangle((0,0,image.width,48),fill='black')
                controller = episode.metadata.get('controller', 'classical_only').replace('_', ' ').upper()
                draw.text((8,6),f'PRERECORDED EVALUATION | {controller}',fill='white')
                label = 'running' if index < len(episode.rows)-1 else ('success' if episode.outcome['success'] else 'failure')
                draw.text((8,25),f"{episode.metadata['embodiment']} | {episode.metadata['target']} | {label}",fill='white')
                frame = av.VideoFrame.from_image(image)
                frame.pts,frame.time_base = index,Fraction(1,30)
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
    return {'prerecorded':True,'controller':'classical_only','frames':len(episode.rows),'output':str(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(render(args.episode,args.output))


if __name__ == '__main__':
    main()
