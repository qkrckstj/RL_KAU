"""GPU weapon geometry and public observations, independently tested against scalar code."""
import math
import torch
from .control import wrap


def engagement(aircraft):
    own, foe = aircraft, aircraft.flip(-2)
    delta = foe[..., :3] - own[..., :3]
    dv = foe[..., 3:6] - own[..., 3:6]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    nose = torch.stack((own[..., 11].sin(), own[..., 11].cos(), own[..., 10].sin()), -1)
    other = nose.flip(-2)

    def angle(a, b):
        an, bn = torch.linalg.vector_norm(a, dim=-1), torch.linalg.vector_norm(b, dim=-1)
        cos = (a*b).sum(-1) / (an*bn).clamp_min(1e-30)
        return torch.where((an < 1e-6) | (bn < 1e-6), 0., cos.clamp(-1,1).acos())

    aim = delta + dv * (distance/1050).unsqueeze(-1)
    ata, lead, aspect = angle(delta,nose), angle(aim,nose), angle(delta,other)
    inside = (lead <= math.pi/6) & (distance >= 150) & (distance <= 1500)
    dmg = (1-lead/(math.pi/6)) * (1-(distance-150)/1350) * (.15 + .85*aspect.cos().clamp_min(0))
    valid = distance >= 1e-6
    z = lambda x: torch.where(valid,x,0.)
    return dict(ata=z(ata), ata_signed=z(wrap(torch.atan2(delta[...,0],delta[...,1])-own[...,11])),
        ata_lead=z(lead), aa=z(aspect), r=distance,
        r_dot=z((delta*dv).sum(-1)/distance.clamp_min(1e-30)), in_wez=inside,
        damage_rate=torch.where(inside,dmg,0.),
        lead_signed=wrap(torch.atan2(aim[...,0],aim[...,1])-own[...,11]))


def observe(aircraft, health, track, wez, remaining):
    boundary = 50000 - torch.linalg.vector_norm(aircraft[..., :2], dim=-1)
    match = torch.stack((health,health.flip(-1),track,track.flip(-1),wez,wez.flip(-1),
        boundary,boundary.flip(-1),remaining[:,None].expand(-1,2)), -1)
    return torch.cat((aircraft,aircraft.flip(-2),match),-1)


def features(x):
    dx,dy,dz = (x[...,15+i]-x[...,i] for i in range(3))
    distance = (dx*dx+dy*dy+dz*dz).sqrt()+1e-9
    horizontal = torch.hypot(dx,dy)
    nx = torch.where(horizontal > 1e-6, dx/horizontal.clamp_min(1e-30),0.)
    ny = torch.where(horizontal > 1e-6, dy/horizontal.clamp_min(1e-30),0.)
    si,co = x[...,11].sin(),x[...,11].cos()
    dvx,dvy = x[...,18]-x[...,3],x[...,19]-x[...,4]
    hd = x[...,26]-x[...,11]
    values = [distance/(distance+10000), (dx*si+dy*co)/(distance+10000),
        (dx*co-dy*si)/(distance+10000),nx*co-ny*si,nx*si+ny*co,hd.sin(),hd.cos(),
        ((dvx*nx+dvy*ny)/400).tanh(),((dvx*ny-dvy*nx)/400).tanh(),
        (torch.linalg.vector_norm(x[...,3:6],dim=-1)/300).tanh(),
        (torch.linalg.vector_norm(x[...,18:21],dim=-1)/300).tanh(),
        x[...,9].sin(),x[...,9].cos(),x[...,24].sin(),x[...,24].cos(),
        (x[...,14]/.5).tanh(),(x[...,29]/.5).tanh(),(x[...,8]/9).tanh(),(x[...,23]/9).tanh(),
        x[...,30],x[...,31],x[...,32].clamp(max=1),x[...,33].clamp(max=1),x[...,34],x[...,35],
        (x[...,36]/50000).tanh(),(x[...,37]/50000).tanh(),x[...,38]/120,
        (dz/1000).tanh(),(x[...,5]/100).tanh(),(x[...,20]/100).tanh()]
    return torch.stack(values,-1).clamp(-1,1)


def ace_actions(aircraft):
    e = engagement(aircraft)
    turn = torch.where(e['lead_signed'].abs() <= math.radians(3.75), 1,
                       torch.where(e['lead_signed'] > 0,2,0))
    speed = torch.where(e['r'] > 900,2,torch.where((e['r_dot'] > -20)&(e['r'] > 400),1,0))
    return turn*3+speed


def reactive_actions(x, parameters):
    p = parameters.unbind(-1)
    duration,vo,vt,lead,bias,brake,dead,direction,stop,far,vfar,dr,da,offset,vd,ea,er = p
    dx,dy=x[...,15]-x[...,0],x[...,16]-x[...,1]
    distance=torch.hypot(dx,dy)
    enemy=wrap(torch.atan2(-dx,-dy)-x[...,26]).abs()*180/math.pi
    opening=(120-x[...,38] < duration)&(distance >= stop)
    opening=opening & ~((er > 0)&(distance > er)&(enemy > ea))
    bearing=torch.atan2(dx+lead*x[...,18],dy+lead*x[...,19])
    demand=(wrap(bearing-x[...,11]+bias*math.pi/180)-brake*x[...,14])*180/math.pi
    steer=lambda a: torch.where(a > dead,2,torch.where(a < -dead,0,1))
    turn=torch.where(opening,torch.where(direction < 0,0,2),steer(demand))
    speed=torch.where(opening,vo,torch.where((far > 0)&(distance > far),vfar,vt))
    defense=(distance < dr)&(enemy < da)&(x[...,34] < .5)
    demand=(wrap(x[...,26]+offset*math.pi/180-x[...,11])-brake*x[...,14])*180/math.pi
    turn=torch.where(defense,steer(demand),turn)
    speed=torch.where(defense,vd,speed)
    own=torch.linalg.vector_norm(x[...,3:6],dim=-1)/.514444
    throttle=torch.where(own < speed-5,2,torch.where(own > speed+5,0,1))
    return 3*turn+throttle
