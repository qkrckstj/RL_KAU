"""Distinct local CEM proposals that perturb one controller block at a time."""
import numpy as np

BLOCKS=((0,1,7,8),(2,3,4,5,6),(11,12,13,14),(9,10,15,16))


def sample_population(rng,mean,std,anchors,size,uniform=1,opening_scale=.05):
    mean,std=np.asarray(mean,float),np.asarray(std,float)
    if (mean.shape!=(17,) or std.shape!=mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std<0) or np.any(mean<0)
            or np.any(mean>1) or uniform<0 or size<1 or not 0<opening_scale<=1):
        raise ValueError('Invalid block proposal distribution')
    population,seen,anchor_indices,origins=[],{},[],[]
    def add(v,origin):
        key=tuple(v)
        if key in seen: return seen[key],False
        i=len(population);seen[key]=i;population.append(v.copy());origins.append(origin)
        return i,True
    for vector in anchors:
        v=np.asarray(vector,float)
        if v.shape!=mean.shape or not np.isfinite(v).all() or np.any(v<0) or np.any(v>1):
            raise ValueError('Invalid anchor')
        index,_=add(v,dict(kind='anchor'));anchor_indices.append(index)
    if len(population)+uniform>size: raise ValueError('Population cannot fit anchors and immigrants')
    sigma=std.copy();sigma[0]*=opening_scale
    misses=0
    while len(population)<size:
        if len(population)>=size-uniform or misses>=32:
            v=rng.random(17); origin=dict(kind='uniform')
        else:
            block=list(BLOCKS[int(rng.integers(len(BLOCKS)))])
            v=mean.copy();v[block]=np.clip(rng.normal(mean[block],sigma[block]),0,1)
            origin=dict(kind='block',indices=block)
        _,added=add(v,origin);misses=0 if added else misses+1
    return np.asarray(population),anchor_indices,origins
