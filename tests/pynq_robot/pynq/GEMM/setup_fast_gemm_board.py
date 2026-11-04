"""Configure the requested existing GEMM_V2 overlay at 250 MHz, clock enabled."""
from pathlib import Path
from pynq import Overlay, MMIO, Clocks, PL

bitfile = Path(__file__).resolve().parent / 'hardware/design_1.bit'
if PL.bitfile_name:
    raise RuntimeError('Overlay already loaded; inspect before reconfiguration')
overlay = Overlay(str(bitfile), download=False)
clocks = Clocks._instance
io = clocks._crl_registers.IOPLL_CTRL
rp = clocks._crl_registers.RPLL_CTRL
pll_values = (int(io[:]), int(rp[:]))
parent_mhz = float(clocks.get_pll_mhz(io))
div0, div1 = min(((a,b) for a in range(1,64) for b in range(1,64)),
    key=lambda pair: (abs(parent_mhz/(pair[0]*pair[1])-250), pair[1]!=1, pair[1]))
assert abs(parent_mhz/(div0*div1)-250)<0.1
overlay.clock_dict[0].update(enable=1, divisor0=div0, divisor1=div1)
control = MMIO(0xFF5E00C0, 4)
original = int(control.read(0))
mask = 7 | (63<<8) | (63<<16) | (1<<24)
configured = (original & ~mask) | (div0<<8) | (div1<<16)
try:
    control.write(0, original & ~(1<<24))
    control.write(0, configured)
    control.write(0, configured | (1<<24))
    assert abs(Clocks.fclk0_mhz-250)<0.1
    print('Clock enabled before download', Clocks.fclk0_mhz, flush=True)
    overlay.download()
    current = int(control.read(0))
    assert current & (1<<24) and (current & 7)==0
    assert abs(Clocks.fclk0_mhz-250)<0.1
    assert (int(io[:]), int(rp[:])) == pll_values
except BaseException:
    control.write(0,original)
    raise
print('Loaded', PL.bitfile_name, 'PL MHz', Clocks.fclk0_mhz, flush=True)
for base in (0x80000000,0x80001000,0x80002000):
    mm = MMIO(base,0x200)
    print(hex(base),'signature',hex(mm.read(0x50)),'status',hex(mm.read(0x18)),flush=True)
    assert mm.read(0x50)==0x0A080020
    assert not (mm.read(0x18)&65)
