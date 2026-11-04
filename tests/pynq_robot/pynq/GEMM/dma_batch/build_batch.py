"""Reuse the previously prepared isolated kernel build; never edit system headers."""
from pathlib import Path
import subprocess

base=Path(__file__).resolve().parent
prepared=base.parent/'dma_trial'
build=prepared/'kernel_build'
compat=prepared/'compat_include'
if not (build/'include/linux/dmaengine.h').exists():
    raise RuntimeError('Run python3 ../dma_trial/build_trial.py first, or use a matching prepared kernel tree')
subprocess.check_call(['make','-C',str(build),'-j4','M='+str(base),
    'CONFIG_STACKPROTECTOR_PER_TASK=','CONFIG_STACKPROTECTOR=',
    'CONFIG_STACKPROTECTOR_STRONG=','KCFLAGS=-I'+str(compat),'modules'])
