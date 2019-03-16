import sys
from time import time

def update_progress(progress, starttime=None):
    barLength = 50 # Modify this to change the length of the progress bar
    total_length = barLength + 16 + 5 + 6 + 3
    status = ""
    if progress == "done":
        sys.stdout.write("\r{}".format(" "*total_length))
        sys.stdout.write("\r")
        sys.stdout.flush()        
        return
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    if (starttime is None) or (progress < 0.025):
        text = "\rPercent: [{0}] {1:5.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    else:
        t = time()
        t_remain = (t-starttime) / progress * (1.0 - progress)
        text = "\rPercent: [{0}] {1:5.2f}% {2:6.1f}s {3}".format( "#"*block + "-"*(barLength-block), progress*100, t_remain, status)
    sys.stdout.write(text)
    sys.stdout.flush()