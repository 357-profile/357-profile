"""Tone-match the outpainted ceiling/floor to the photo, add a soft natural falloff, upscale x2 and size to 1080x1920."""
import json, numpy as np, cv2, torch
from PIL import Image
from spandrel import ModelLoader
torch.set_num_threads(4)
up=ModelLoader().load_from_file('/tmp/claude-0/-home-user-357-profile/6bba53e8-0a57-5a08-91ac-769c623010ed/scratchpad/models/RealESRGAN_x2plus.pth').eval()
def esr(a):
    t=torch.from_numpy(a.astype(np.float32)/255.).permute(2,0,1)[None]
    with torch.no_grad(): o=up(t)
    return (o[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
cfg=json.load(open('work/vert_wide.json'))
log=open('work/vertw/ext.txt').read().split('\n') if False else None
for n,c in cfg.items():
    A=np.array(Image.open(f'work/vertw/{n}_after_raw.png')); B=np.array(Image.open(f'work/vertw/{n}_before_raw.png')); M=np.array(Image.open(f'work/vertw/{n}_mask_raw.png'))
    H=A.shape[0]; rows=np.where((A!=B).any(2).any(1) | (M>0).any(1))[0]
    # band = original photo rows; recover from where before/after/mask could differ is unreliable -> recompute from config
    src=np.array(Image.open(c['after']).convert('RGB')); h0=src.shape[0]
    if c.get('pre_scale'): h0=round(h0*c['pre_scale'])
    ext=H-h0; top=round(ext*c.get('top_share',0.45)); bot=ext-top
    def fix(img):
        out=img.astype(np.float32); lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB).astype(np.float32)
        for (r0,r1),(s0,s1),far in (((0,top),(top,top+40),0),((H-bot,H),(H-bot-40,H-bot),H-1)):
            if r1<=r0: continue
            ref=lab[s0:s1].reshape(-1,3).mean(0); reg=lab[r0:r1]
            near=reg[(r1-r0-24 if far==0 else 0):(r1-r0 if far==0 else 24)].reshape(-1,3).mean(0)
            reg[...,1]+=ref[1]-near[1]; reg[...,2]+=ref[2]-near[2]; reg[...,0]+=(ref[0]-near[0])*0.7
            lab[r0:r1]=reg
        out=cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2RGB).astype(np.float32)
        y=np.arange(H,dtype=np.float32)
        g=np.ones(H,np.float32)
        if top>0: t=np.clip((top-y)/top,0,1); g*=1-0.38*t**1.4      # ceiling gently darker away from the lights
        if bot>0: t=np.clip((y-(H-bot))/bot,0,1); g*=1-0.30*t**1.3   # foreground floor slightly darker
        out*=g[:,None,None]
        # keep the original photo band untouched
        out[top:H-bot]=img[top:H-bot]
        # soften the seam
        for y0,d in ((top,-1),(H-bot,1)):
            for i in range(10):
                yy=y0+(i if d==1 else -1-i); a=(i+1)/11
                if 0<=yy<H: out[yy]=out[yy]*a+cv2.GaussianBlur(out,(1,5),0)[yy]*(1-a) if False else out[yy]
        return np.clip(out,0,255).astype(np.uint8)
    A2,B2=fix(A),fix(B)
    A3=cv2.resize(esr(A2),(1080,1920),interpolation=cv2.INTER_AREA); B3=cv2.resize(esr(B2),(1080,1920),interpolation=cv2.INTER_AREA)
    M3=cv2.resize(M,(1080,1920),interpolation=cv2.INTER_LINEAR)
    for k,v in (('after',A3),('before',B3)): Image.fromarray(v).save(f'work/vertw/{n}_{k}.png')
    Image.fromarray(M3).save(f'work/vertw/{n}_mask.png'); print(n,'top',top,'bot',bot,flush=True)
