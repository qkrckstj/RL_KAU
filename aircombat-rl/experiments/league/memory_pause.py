"""Wait without advancing a simulation when the existing memory guard fires."""
import time

GUARD_MESSAGE='Commit/physical headroom below safe operating floor'


def guarded_step(step,waiter):
    """The existing callback has already counted the transition: never repeat it."""
    try:
        return step()
    except MemoryError as error:
        if str(error)!=GUARD_MESSAGE:
            raise
        waiter()
        return True


def wait_for_headroom(read_resources,emit,*,clock=time.monotonic,sleep=time.sleep,
                      max_seconds=300.,interval=5.,resume_commit=2.8,resume_physical=2.):
    started=clock()
    healthy=0
    samples=0
    minimum=float('inf')
    while True:
        resources=read_resources()
        elapsed=clock()-started
        samples+=1
        minimum=min(minimum,resources['commit_headroom_gib'])
        row=dict(elapsed_seconds=elapsed,samples=samples,resources=resources,
                 minimum_commit_headroom_gib=minimum)
        if resources['commit_headroom_gib']<.5 or resources['available_memory_gib']<.75:
            emit(dict(stage='resource_pause_critical_abort',**row))
            raise MemoryError('Critical memory floor while paused')
        healthy=healthy+1 if (resources['commit_headroom_gib']>=resume_commit and
                              resources['available_memory_gib']>=resume_physical) else 0
        if healthy>=2:
            emit(dict(stage='resource_pause_recovered',**row))
            return row
        if elapsed>=max_seconds:
            emit(dict(stage='resource_pause_timeout',**row))
            raise MemoryError('Memory headroom did not recover before timeout')
        if samples==1 or (samples-1)%3==0:
            emit(dict(stage='resource_pause_waiting',**row))
        sleep(min(interval,max_seconds-elapsed))
