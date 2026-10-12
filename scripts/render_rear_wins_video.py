"""Render recorded simulator trajectories to MP4; does not simulate or edit policies."""
import json,math,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'video_tools'))
import imageio_ffmpeg

import argparse
from functools import lru_cache
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
OUT=args.out.resolve()
if (OUT/'rear_wins.mp4').exists():raise FileExistsError(OUT/'rear_wins.mp4')
DATA=json.loads((OUT/'trajectories.json').read_text('utf-8'))
FONT='C:/Windows/Fonts/malgun.ttf'
@lru_cache(maxsize=16)
def font(n):return ImageFont.truetype(FONT,n)
BG='#101925';GRID='#273446';FG='#ebf1f8';MUTED='#a8b8cb'
def render(clip,i,ordinal,total):
    fs=clip['frames'];f=fs[i];im=Image.new('RGB',(1280,720),BG);d=ImageDraw.Draw(im)
    foe=clip['foe_label']
    owncolor='#f86b76' if clip['seat']=='red' else '#62adff'
    enemycolor='#62adff' if clip['seat']=='red' else '#f86b76'
    d.text((28,18),f'{clip["label"]}  vs  {foe}',font=font(27),fill=FG)
    d.text((30,60),f'{ordinal}/{total} 경기  ·  자기 진영: {clip["seat"].upper()}  ·  2배속  ·  고정 고도',font=font(18),fill=MUTED)
    box=(28,105,865,659);cx=(f['own_x']+f['opp_x'])/2;cy=(f['own_y']+f['opp_y'])/2
    span=max(2200,abs(f['own_x']-f['opp_x'])*1.5,abs(f['own_y']-f['opp_y'])*2.2)
    scale=(box[2]-box[0])/span
    def xy(x,y):return ((box[0]+box[2])/2+(x-cx)*scale,(box[1]+box[3])/2-(y-cy)*scale)
    arena=Image.new('RGB',(box[2]-box[0],box[3]-box[1]),'#152232');a=ImageDraw.Draw(arena)
    def local(x,y):u,v=xy(x,y);return u-box[0],v-box[1]
    grid=1000 if span>6000 else 500
    for x in range(int((cx-span)/grid)*grid,int((cx+span)/grid)*grid+grid,grid):
        u,_=local(x,cy);a.line((u,0,u,arena.height),fill=GRID)
    for y in range(int((cy-span)/grid)*grid,int((cy+span)/grid)*grid+grid,grid):
        _,v=local(cx,y);a.line((0,v,arena.width,v),fill=GRID)
    for side,color in [('own',owncolor),('opp',enemycolor)]:
        pts=[local(r[side+'_x'],r[side+'_y']) for r in fs[max(0,i-100):i+1]]
        if len(pts)>1:a.line(pts,fill=color,width=2)
    if i:
        prev=fs[i-1]
        for victim,shooter in [('own','opp'),('opp','own')]:
            if f[victim+'_health']<prev[victim+'_health']-1e-7:
                a.line((*local(f[victim+'_x'],f[victim+'_y']),*local(f[shooter+'_x'],f[shooter+'_y'])),fill='#ffd67c',width=3)
    for side,color in [('own',owncolor),('opp',enemycolor)]:
        u,v=local(f[side+'_x'],f[side+'_y']);psi=f[side+'_psi']
        forward=(math.sin(psi),-math.cos(psi));right=(math.cos(psi),math.sin(psi))
        def pt(fw,rt):return (u+fw*forward[0]+rt*right[0],v+fw*forward[1]+rt*right[1])
        a.polygon([pt(18,0),pt(-10,11),pt(-5,0),pt(-10,-11)],fill=color)
    im.paste(arena,box[:2]);d.rectangle(box,outline=GRID,width=2)
    d.text((46,117),f'격자 {grid:,}m  ·  최근 20초 궤적',font=font(15),fill=MUTED)
    d.text((905,110),f'{f["t"]:05.1f} / 120초',font=font(30),fill=FG)
    distance=math.hypot(f['own_x']-f['opp_x'],f['own_y']-f['opp_y'])
    d.text((905,162),f'기체 간 거리  {distance:,.0f}m',font=font(19),fill=MUTED)
    for k,label,color,y in [('own',clip['label'],owncolor,225),('opp','상대 정책',enemycolor,325)]:
        hp=max(0,min(1,f[k+'_health']))
        d.text((905,y),label,font=font(21),fill=color)
        d.text((1130,y),f'{100*hp:.1f}%',font=font(21),fill=FG)
        d.rectangle((905,y+37,1240,y+56),fill=GRID)
        if hp>0:d.rectangle((905,y+37,905+335*hp,y+56),fill=color)
    d.text((905,403),f'후방에서 준 피해 {100*f["rear_damage"]:.1f}%',font=font(20),fill='#74e6a2')
    if f['rear_firing']:
        d.rounded_rectangle((48,605,260,645),radius=8,fill='#215b3d')
        d.text((62,611),'후방 명중 중',font=font(22),fill='#c7ffdd')
    d.text((905,443),'종료 시 남은 체력으로 판정',font=font(18),fill=FG)
    d.text((905,473),'소수점 4자리 반올림 · 동률 무승부',font=font(15),fill=MUTED)
    if i==len(fs)-1:
        result={'WIN':'자기 승리','LOSS':'자기 패배','DRAW':'무승부'}[clip['result']]
        d.rounded_rectangle((900,527,1246,605),radius=12,fill='#253b52')
        d.text((928,544),result,font=font(31),fill=FG)
    else:d.text((905,550),'경기 진행 중',font=font(24),fill=MUTED)
    d.text((30,680),f'실제 JSBSim 기록 재생 · seed {clip["seed"]} · 원래 FairFight 시작 · 선택한 성공 사례 · 일반 승률을 뜻하지 않음',font=font(16),fill=MUTED)
    return im

writer=imageio_ffmpeg.write_frames(str(OUT/'rear_wins.mp4'),(1280,720),fps=10,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=8,output_params=['-movflags','+faststart'])
writer.send(None);count=0
try:
    for n,clip in enumerate(DATA['clips'],1):
        for i in range(len(clip['frames'])):
            im=render(clip,i,n,len(DATA['clips']))
            if n==1 and clip['frames'][i]['rear_firing'] and not (OUT/'preview.png').exists():im.save(OUT/'preview.png')
            writer.send(im.tobytes());count+=1
        for _ in range(25):writer.send(im.tobytes());count+=1
finally:writer.close()
(OUT/'video_manifest.json').write_text(json.dumps(dict(frames=count,fps=10,duration_seconds=count/10,speed=2,clips=[{k:c[k] for k in ('foe','seat','seed','result','own_health','opp_health')} for c in DATA['clips']]),indent=2),'utf-8')
print('video_complete',count/10,str(OUT/'rear_wins.mp4'))
