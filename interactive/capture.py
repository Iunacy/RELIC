import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('execution',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=json.loads((args.execution/'comparison.json').read_text())
    if args.output.exists():
        p.exit(2,'output exists; choose a fresh path\n')
    if not shutil.which('ffmpeg'):
        p.exit(2,'ffmpeg is required for MP4 capture\n')
    work=args.execution/'capture'
    work.mkdir(exist_ok=False)
    for control in report['controllers']:
        method=control['controller']
        frames=control.get('frames',[])
        if not frames:
            p.exit(2,'no observed-state frames; rerun demo without --no-capture\n')
        metrics=control['metrics']
        label=f'{report["robot"]} | {method} | {control["status"]} | observed-state replay'
        label+=f' | RMSE {metrics["rmse_rad"]:.5f} rad | final {metrics["final_m"]*1000:.2f} mm'
        (work/(method+'.txt')).write_text(label)
        source=args.execution/Path(frames[0]).parent/'frame-%05d.jpg'
        duration=metrics['simulation_s']
        fps=(len(frames)-1)/duration
        subprocess.run(['ffmpeg','-v','error','-n','-framerate',str(fps),'-i',str(source),
                        '-vf',f"drawtext=textfile='{work/method}.txt':fontsize=12:fontcolor=white:box=1:boxcolor=black@0.65:x=10:y=h-30,pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        '-c:v','libx264','-pix_fmt','yuv420p',str(work/(method+'.mp4'))],check=True)
    listing=work/'clips.txt'
    listing.write_text(''.join("file '"+str((work/(c['controller']+'.mp4')).resolve())+"'\n" for c in report['controllers']))
    subprocess.run(['ffmpeg','-v','error','-n','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(args.output)],check=True)
    print(args.output)


if __name__=='__main__':
    main()
