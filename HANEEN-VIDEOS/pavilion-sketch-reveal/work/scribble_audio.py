"""Build the scribble sound track from the reference reel: one scribble cycle per scene, its reveal 'tick' landing on our reveal frame."""
import numpy as np, subprocess, json, wave
sr=48000
raw=subprocess.run(['ffmpeg','-v','error','-i','audio/ref_audio.wav','-f','f32le','-ac','2','-ar',str(sr),'-'],capture_output=True).stdout
ref=np.frombuffer(raw,np.float32).reshape(-1,2)
ticks=[2.160,4.945,7.710,10.645]          # reveal ticks found in the reference audio
order=[0,1,2,3,1,2]                         # which reference cycle each of our 6 scenes uses
cfg=json.load(open('work/scenes.json')); T=cfg['timing']
scene_len=[sc.get('timing',{}).get('frames',T['frames'])/30 for sc in cfg['scenes']]
boom=T['boom']/30
out=np.zeros((int(round(sum(scene_len)*sr)),2),np.float32); t0=0.0
def fade(seg,fi,fo):
    n=len(seg); a=np.ones(n,np.float32); i=int(fi*sr); o=int(fo*sr)
    if i: a[:i]=np.linspace(0,1,i)
    if o: a[n-o:]=np.linspace(1,0,o)
    return seg*a[:,None]
for k,L in enumerate(scene_len):
    tk=ticks[order[k]]; a=tk-boom; b=min(tk+0.55,len(ref)/sr)
    seg=fade(ref[int(a*sr):int(b*sr)].copy(),0.015,0.08)
    s=int(t0*sr); out[s:s+len(seg)]+=seg[:len(out)-s]; t0+=L
peak=np.abs(out).max(); out=out*min(1.0,0.89/peak)
with wave.open('audio/scribble_track.wav','wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes((np.clip(out,-1,1)*32767).astype('<i2').tobytes())
print('track',len(out)/sr,'s, peak',round(float(np.abs(out).max()),3))
