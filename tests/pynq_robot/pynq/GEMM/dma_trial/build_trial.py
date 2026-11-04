"""Build the trial without modifying the installed kernel/header tree."""
from pathlib import Path
import shutil
import subprocess


def run(*args, **kwargs):
    subprocess.check_call(list(args), **kwargs)


def main():
    base = Path(__file__).resolve().parent
    source = Path('/usr/src/kernel')
    build = base / 'kernel_build'
    compat = base / 'compat_include'
    if not build.exists():
        shutil.copytree(str(source), str(build), symlinks=False)
    for api in ['', 'uapi/']:
        output = compat / api / 'asm'
        output.mkdir(parents=True, exist_ok=True)
        names = set()
        for relative in ['arch/arm64/include/'+api+'asm/Kbuild',
                         'include/'+api+'asm-generic/Kbuild']:
            for line in (source/relative).read_text().splitlines():
                if line.startswith(('generic-y', 'mandatory-y')):
                    names.update(line.split('+=', 1)[1].split())
        for name in names:
            if not (source/('arch/arm64/include/'+api+'asm')/name).exists():
                (output/name).write_text('#include <asm-generic/'+name+'>\n')
    # Root-level fallback includes for missing arch-generated UAPI headers.
    for p in (compat/'uapi/asm').glob('*.h'):
        (compat/'asm'/p.name).write_text(p.read_text())
    run('gcc', '-O2', 'scripts/basic/fixdep.c', '-o',
        'scripts/basic/fixdep', cwd=str(build))
    flags = ('-D__KERNEL__ -mgeneral-regs-only -I./arch/arm64/include '
             '-I./include -I./arch/arm64/include/uapi -I./include/uapi '
             '-I./include/generated/uapi -I'+str(compat)+' '
             '-include ./include/linux/kconfig.h')
    run('make', '-f', 'scripts/Makefile.build', 'obj=scripts/mod',
        'srctree=.', 'CONFIG_SHELL=/bin/sh', 'HOSTCC=gcc', 'CC=gcc',
        'LD=ld', 'ARCH=arm64', 'SRCARCH=arm64', 'KBUILD_CFLAGS='+flags,
        cwd=str(build))
    run('make', '-C', str(build), '-j4', 'M='+str(base),
        'CONFIG_STACKPROTECTOR_PER_TASK=', 'CONFIG_STACKPROTECTOR=',
        'CONFIG_STACKPROTECTOR_STRONG=', 'KCFLAGS=-I'+str(compat), 'modules')


if __name__ == '__main__':
    main()
