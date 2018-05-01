from stft import *
from pvoc import *
from numpy import *


stft_defs = {
        10: STFT(1024, 2, 2),
        11: STFT(2048, 2, 2),
        12: STFT(4096, 2, 2),
        14: STFT(16384, 2, 2),
        16: STFT(65536, 2, 2)
}


time = 0


def f1(time, t, idx, x):
    t //= 9
    y = sin((t|t>>13|t>>9|t>>4|t>>16)*min(time*0.01, 2))/fmax((1+idx[1]), 8)*16
    x.real[:] += y
    x.imag[:] += y*2.2
def f2(time, t, idx, x):
    x.real[:] += sin(t/(1+t%(1+(t//9&t//89^t>>9^t>>13)))*min(time*0.001, 2))/fmax((1+idx[1]), 16)*32
    x.imag[:] += x.real*sin(t*0.01)
def f3(time, t, idx, x):
    y =  sin(idx[0]+(t/(1+((t//5&t//11))%(8192)))*min(time*0.1, 20))/fmax((1+idx[1]**1.5), 1)*8
    y += sin((t/(1+((t//5&t//10))%(8192)))*min(time*0.1, 0.01))/fmax((1+idx[1]**1.0), 1)*2
    x.real[:] += y
    x.imag[:] += y
# cabinet simulator: gain 20, event off-axis, 100hz++
def f4(time, t, idx, x):
    x.real[:] += sin(5*(t%(1+t/(1+(t&t>>9|t>>13^t>>18)))))/fmax((1+idx[1]), 1)*64
    x.imag[:] += cos(4*(t%(1+t/(1+(t&t>>9|t>>12^t>>17)))))/fmax((1+idx[1]), 1)*64

def f5(time, t, idx, x):
    t *= 1
    z = t/(1+t%(1+(t>>5&t>>9^t>>13))+idx[0])
    y = sin(z)/fmax((1+idx[1]**1.0), 1)*8 + 1j*z
    x += from_polar(y)
fns = {
        10: [],
        12: [f5],
        14: [f5],
        16: [f5],
}
time=0

time=0
def f6(time, t, idx, x):
    t *= 1
    z = t%(1+t/(1+(t//(16+(t>>13&t>>16))^t>>10))+idx[0])
    y = sin(z)/fmax((1+idx[1]**1.0), 1)*8 + 1j*sin(idx[1]**0.5*(time)*0.001)*6.28
    x += from_polar(y)
fns = {
        10: [],
        11: [f6],
        12: [],
        14: [],
        16: [],
}

def process(i, o):
    global time
    o[:] = i
    for size in stft_defs:
        stft = stft_defs[size]
        for x in stft.forward(o):
            idx = indices(x.shape)
            t = (time*x.shape[1]+idx[1])
            for fn in fns[size]:
                fn(time, copy(t), idx, x)
            stft.backward(x)
            time += 1
        stft.pop(o)
