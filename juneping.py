from stft import *
from pvoc import *
from numpy import *
def tri(x):
    return abs((x%2*pi)-pi)/pi
def squ(x):
    return sign((x%2*pi)-pi)


stft_defs = {
        8: STFT(256, 2, 2),
        10: STFT(1024, 2, 2),
        11: STFT(2048, 2, 2),
        12: STFT(4096, 2, 2),
        13: STFT(8192, 2, 2),
        14: STFT(16384, 2, 2),
        15: STFT(32768, 2, 2),
        16: STFT(65536, 2, 2),
        17: STFT(65536*2, 2, 2)
        }
stft_defs2 = {
        8: STFT(256, 2, 2),
        10: STFT(1024, 2, 2),
        11: STFT(2048, 2, 2),
        12: STFT(4096, 2, 2),
        13: STFT(8192, 2, 2),
        14: STFT(16384, 2, 2),
        15: STFT(32768, 2, 2),
        16: STFT(65536, 2, 2),
        17: STFT(65536*2, 2, 2)
        }


pvoc_defs = {}
pvoc_defs2 = {}
def gen_pvocs():
    for size in stft_defs:
        pvoc_defs[size] = PhaseVocoder(stft_defs[size])
def gen_pvocs2():
    for size in stft_defs2:
        pvoc_defs2[size] = PhaseVocoder(stft_defs2[size])
gen_pvocs()
gen_pvocs2()
def pvocwrap(x):
    return (x,)


def f24(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = (t&t>>9|t>>13)
    x.real[:] = sin(idx[0]*idx[1]+z/(1+ph)+ph/16)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/512
    x[:] = pv.shift(x, lambda fq: (fq*ph)/4192%44100)
fadeto({
    10: [pvocwrap(f24)],
    })

def f19(time, t, idx, x):
    z =  t*(1/(1+(t>>8&t>>12&t>>15+idx[0])))
    y = squ(z+idx[1])/fmax((1+idx[1]**1.0), 1)*x.shape[1]/2048 + 1j*squ(z*idx[1]+idx[0])*2*pi
    x += from_polar(y)
fadeto({
    10: [],
    11: [],
    12: [],
    14: [f19],
    16: [f19],
    17: [f19],
    })

fadeto({
    10: [f19],
    11: [],
    12: [],
    14: [f19],
    16: [f19],
    17: [f19],
    })

f1ly=None
def f1(time, t, idx, x):
    global f1ly
    if f1ly is None:
        f1ly=zeros(x.shape)
    t //= 9
    y = sin((t|t>>13|t>>9|t>>4|t>>16^idx[0])*min(time*0.01, 2))/fmax((1+idx[1]), 8)*128
    y = where(f1ly>y, f1ly*0.9, y*0.8+f1ly*0.2)
    f1ly[:] = y
    x.real[:] += y
    x.imag[:] += y*2.2
fadeto({
    14: [f1],
    })

def f2(time, t, idx, x):
    x.real[:] += sin(t/(1+t%(1+(t//9&t//89^t>>9^t>>13)))*min(time*0.001, 2))/fmax((1+idx[1]), 16)*16
    x.imag[:] += x.real*sin(t*0.01)
fadeto({
    14: [f2],
    })

def f4(time, t, idx, x):
    x.real[:] += sin(5*(t%(1+t/(1+(t&t>>9|t>>13^t>>18)))))/fmax((1+idx[1]), 1)*8
    x.imag[:] += cos(4*(t%(1+t/(1+idx[0]+(t&t>>9|t>>12^t>>17)))))/fmax((1+idx[1]), 1)*8
fadeto({
    10: [],
    11: [],
    12: [],
    14: [f4],
    16: [f4],
    17: [],
    })

def f3(time, t, idx, x):
    t = (time*7*x.shape[1]+idx[1])
    y =  sin(idx[0]+(t/(1+((t//5&t//11))%(8192)))*min(time*0.1, 20))/fmax((1+idx[1]**1.5), 1)*16
    y += sin((t/(1+((t//5&t//10))%(8192)))*min(time*0.1, 0.01))/fmax((1+idx[1]**1.0), 1)*6
    x.real[:] += y
    x.imag[:] += y
fadeto({
    12: [f3],
    16: [f4],
    })


def f6(time, t, idx, x):
    t = (time*7*x.shape[1]+idx[1])
    t *= 1
    z = t%(1+t/(1+(t//(16+(t>>13&t>>16))^t>>10))+idx[0])
    y = sin(z)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/256 + 1j*sin(idx[1]**0.5*(time)*0.005)*1
    x += from_polar(y)
fadeto({
    10: [],
    11: [f6],
    12: [],
    14: [],
    16: [],
    17: [],
    })

def f7(time, t, idx, x):
    t = (time*1*x.shape[1]+idx[1])
    t *= int(time/256)+1
    z = idx[0]+ t*(1/(1+(t//4096&t//8192)%256))
    y = sin(z/256)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/384 + 1j*sin(idx[1]*z)*1
    x += from_polar(y)
fadeto({
    10: [],
    11: [f6,f7],
    12: [],
    14: [],
    16: [],
    17: [],
    })

def f8(time, t, idx, x):
    t = (time*7*x.shape[1]+idx[1])
    t *= int(time/2)+1
    z = idx[0]+ t*(1/(1+(t>>6^t>>13^t>>14)))
    y = sin(z)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/1024 + 1j*sin(idx[1]*z)*1
    x += from_polar(y)
fadeto({
    10: [],
    11: [],
    12: [],
    13: [],
    14: [],
    16: [f8],
    17: [],
    })

def f9(time, t, idx, x):
    t = (time*7*x.shape[1]+idx[1])
    z = idx[0]+ t*(1/(1+(t>>1&t>>2)))
    y = sin(z)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/1024 + 1j*sin(idx[1]*z)*1
    x += from_polar(y)
fadeto({
    10: [],
    11: [],
    12: [],
    14: [],
    16: [],
    17: [f9],
    })

def f10p1(time, t, idx, x):
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0]+ t/((1+(t>>8&t>>10&t>>12)))
    y = sin(z/(idx[1]+1))/fmax((1+idx[1]**1.0), 1)*x.shape[1]/1024 + 1j*sin(z*z)*1
    x += from_polar(y)
def f10(time, t, idx, x):
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0]+ t%((1+(t>>8&t>>10&t>>12)))
    y = sin(z/(idx[1]+1))/fmax((1+idx[1]**1.0), 1)*x.shape[1]/1024 + 1j*sin(idx[1]+z/(1+idx[1]))*1
    x += from_polar(y)

fadeto({
    10: [f10],
    11: [f10],
    12: [],
    14: [f10,f10p1],
    16: [],
    17: [f10],
    })

fadeto_cont({
    10: [f10],
    11: [f10],
    14: [f10],
    16: [f10],
    17: [f10],
    })

def f12(time, t, idx, x):
    t += 1024*1024
    z = 1+t|(t*(1/(1+(idx[0]*32+t>>8&t>>12&t>>15)))).astype(int)
    y = squ(squ(t//16384)+z*z)/fmax(abs((idx[1]**1.0)), 1)*x.shape[1]/1024 + 1j*squ(idx[1]*z)*2*pi
    x += from_polar(y)
fadeto({
    10: [f12],
    })


def f11(time, t, idx, x):
    t = (time*7*x.shape[1]+idx[1])
    z = idx[0]+ t*(1/(1+(t>>4&t>>16)))
    y = tri(z*256)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/1024 + 1j*idx[1]*z
    x += from_polar(y)
fadeto({
    10: [f11],
    16: [f11],
    })


prev=None
def f13(time, t, idx, x):
    t = (time*9*x.shape[1]+idx[1])
    global prev
    if prev is None:
        prev = zeros(x.shape).astype(int)
    z = idx[0] + t/(1+(prev>>9&t>>9&t>>12^t>>13))
    prev = z.astype(int)*t
    y = sin(z+idx[1]**0.9)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/1024 + 1j*(1+idx[1])/(1+prev)
    x += from_polar(y)*0.5
f14p=None
def f14(time, t, idx, x):
    t = (time*9*x.shape[1]+idx[1])
    global f14p
    if f14p is None:
        f14p = zeros(x.shape).astype(int)
    z = t/(1+((f14p>>12^f14p>>9)^t>>13))
    f14p = (f14p*0.5 + z*t*0.5+idx[0]).astype(int)
    y = sin(z)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/512 + 1j*((idx[1]**1.005)*z)
    x += from_polar(y)*0.5
fadeto({
    10: [],
    11: [],
    12: [],
    14: [f14],
    16: [f13],
    17: [],
    })

def f25(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = t
    ph = ((t^t>>2^t>>4^t>>7^t>>11^t>>14)%((1<<16)+idx[1]))*(idx[1])+idx[0]
    x[:] = pv.shift(x, lambda fq: (ph)%22050)
    x.real[:] = sin(z/(1+ph))/fmax((1+idx[1]**0.8), 1)*x.shape[1]/1024
    ph = ((t>>7&t>>11&t>>14))+idx[0]*idx[1]*0.001
    pv.shift(x, lambda fq: (ph)%22050).real
fadeto({
    11: [pvocwrap(f25)],
    })


def f15(time, _, idx, x):
    t = (time*7*x.shape[1]+idx[1])
    z = idx[0] + (t&(1+(t>>8|t>>12|t>>15)))
    ph = pi/(1+(t>>13|t>>15|t>>18))
    y = sin(z*z)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/512 + 1j*(ph)
    x += from_polar(y)
fadeto({
    10: [],
    11: [],
    12: [f15],
    14: [],
    16: [],
    17: [],
    })

def f16(time, t, idx, x):
    ph = (t>>3^t>>5^t>>7^idx[0])%16384
    mf = (((time*(time//4|time//16))&7))
    y = (mf>0)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/1024 - 1j*(0.01*mf*ph*idx[1]**0.05)
    x += from_polar(y)
rnd=None
ph=None
def f17(time, t, idx, x):
    global rnd
    global ph
    if rnd is None:
        ph = random.permutation(idx[1])
        rnd = random.random(x.shape)
    mf = ((((time//32&time//17))&7))
    y = (mf>0)*sin(ph**0.01*time*mf)*rnd/fmax((1+idx[1]**0.8), 1)*x.shape[1]/1024 - 1j*sin(mf+idx[0])
    x += from_polar(y)*0.5
def f18(time, t, idx, x):
    t *= int(1+time/1)
    z = idx[0]+ t*(1/(1+(t>>13^t>>16)))
    y = sin(z)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/1024 + 1j*sin(idx[1]*z)*1
    x += from_polar(y)*0.5
fadeto({
    10: [f17],
    11: [],
    12: [],
    13: [f16],
    14: [],
    16: [],
    17: [f18],
    })

def f20(time, _, idx, x, pv):
    time += 1000 # try 10000. 0
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = log1p((t>>4^t>>7^t>>11))*4096
    x[:] = sin(z*ph)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/256 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: (ph*fq)/16384%22050)
fadeto({
    12: [pvocwrap(f20)],
    })

def f21(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = log1p((t&t//30|t//632|t//4050)**2.5)*512
    x[:] = sin(z/256*pi)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/256 + 1j*idx[1]/idx[1].shape[1]*22050
    x[:] = pv.shift(x, lambda fq: idx[0]+(ph*fq)/8192%22050)
fadeto({
    10: [pvocwrap(f21)],
    })

def f23(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = (t&t>>4|t>>9|t>>13)
    x[:] = pv.shift(x, lambda fq: (ph)%44100)
    x.real[:] = sin(idx[0]*idx[1]+z/(1+ph)+ph/16)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/512
fadeto({
    11: [pvocwrap(f23)],
    })

def f26(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = t
    ph = (t^t>>3^t>>6^t>>10)+idx[0]
    x[:] = pv.shift(x, lambda fq: (ph)%22050)
    x.real[:] = sin(z/(1+ph))/fmax((1+idx[1]**0.9), 1)*x.shape[1]/2048
fadeto({
    15: [pvocwrap(f26)],
    })

# the less interesting pounding bass one
def f22(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = idx[0] + t
    ph = (t|t>>7&t>>9|t>>13)+idx[0]
    x[:] = pv.shift(x, lambda fq: (ph)%44100)
    x.real[:] = sin(z**1.8/(1+ph)+ph/16)/fmax((1+idx[1]**1.0), 1)*x.shape[1]/512 #s/1.8/1.85/1.9/2.0
fadeto({
    16: [pvocwrap(f22)],
    })

def f27(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = t
    ph = ((t^t>>2^t>>4^t>>7^t>>11^t>>14)%((1<<15)+idx[1]))*(idx[1])+idx[0]
    x[:] = pv.shift(x, lambda fq: (ph)%22050)
    x.real[:] = sin(z/(1+ph))/fmax((1+idx[1]**0.7), 1)*x.shape[1]/1024
    ph = ((t>>5^t>>10^t>>15))^idx[0]
    pv.shift(x, lambda fq: (ph)%22050).real
fadeto({
    13: [pvocwrap(f27)],
    })

def f28(time, _, idx, x, pv):
    t = (time*1*x.shape[1]+idx[1])
    z = t
    ph = ((t^t>>2^t>>4^t>>7^t>>11^t>>14)%((1<<15)+idx[1]))*(idx[1])+idx[0]
    x[:] = pv.shift(x, lambda fq: (ph)%22050)
    x.real[:] = sin(z/(1+ph))/fmax((1+idx[1]**0.9), 1)*x.shape[1]/1024
    ph = ((t&t>>5&t>>10&t>>15))^idx[0]
    pv.shift(x, lambda fq: (ph)%22050).real
fadeto({
    10: [pvocwrap(f28)],
    })

fademax = 500
fader = 0 # 0 = fns, fademax = fadefns
fadefns = {}
fadetime = 0
direction = 1
def fadeto(fn):
    global fns
    global fadefns
    global fader
    global direction
    global time
    global fadetime
    direction *= -1
    if direction == -1:
        fns = fn
        time = 0
        gen_pvocs()
    else:
        fadefns = fn
        fadetime = 0
        gen_pvocs2()
    print(fns, fadefns)
def fadeto_cont(fn):
    global fns
    global fadefns
    global fader
    global direction
    global time
    global fadetime
    direction *= -1
    if direction == -1:
        fns = fn
    else:
        fadefns = fn
    print(fns, fadefns)

def process(i, o):
    o[:] = 0
    global time
    global fadetime
    global fader
    fader = min(fademax, max(0, fader + direction))
    fo = copy(o)
    for size in stft_defs:
        if fader != fademax and size in fns and len(fns[size]) > 0:
            stft = stft_defs[size]
            for x in stft.forward(o):
                idx = indices(x.shape)
                t = (time*x.shape[1]+idx[1])
                for fn in fns[size]:
                    if type(fn) == tuple:
                        pv = pvoc_defs[size]
                        x[:] = pv.forward(x)
                        fn[0](time, copy(t), idx, x, pv)
                        x[:] = pv.backward(x)
                    else:
                        fn(time, copy(t), idx, x)
                x *= (fademax - fader)/fademax
                stft.backward(x)
                time += 1
            stft.pop(o)
        if fader != 0 and size in fadefns and len(fadefns[size]) > 0:
            stft = stft_defs2[size]
            for x in stft.forward(fo):
                idx = indices(x.shape)
                t = (fadetime*x.shape[1]+idx[1])
                for fn in fadefns[size]:
                    if type(fn) == tuple:
                        pv = pvoc_defs2[size]
                        x[:] = pv.forward(x)
                        fn[0](fadetime, copy(t), idx, x, pv)
                        x[:] = pv.backward(x)
                    else:
                        fn(fadetime, copy(t), idx, x)
                x *= fader/fademax
                stft.backward(x)
                fadetime += 1
            stft.pop(fo)
    o += fo



def f5(time, t, idx, x):
    sz = [3,4,0,5][int(log2(x.shape[1])-12)]
    t = (int(time*5+sz)*x.shape[1]+idx[1])
    t *= 1
    z = t/(1+t%(1+(t>>5&t>>9^t>>13))+idx[0])
    y = sin(z)/fmax((1+idx[1]**1.0), 1)*8 + 1j*z
    x += from_polar(y)
fns = {
        10: [],
        11: [],
        12: [],
        17: [],
        13: [f5],
        14: [f5],
        16: [f5],
        }
time=0
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
