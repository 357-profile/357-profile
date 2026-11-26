"""Wider 9:16 framing: crop wider than the photo height allows, then extend ceiling/floor with LaMa."""
import sys, json, numpy as np, torch, cv2
from PIL import Image
torch.set_num_threads(4)
lama=torch.jit.load('/tmp/claude-0/-home-user-357-profile/6bba53e8-0a57-5a08-91ac-769c623010ed/scratchpad/models/big-lama.pt',map_location='cpu').eval()
def load(p): return np.array(Image.open(p).convert('RGB'))
def run_lama(img, mask):
    H,W=mask.shape; ph=(8-H%8)%8; pw=(8-W%8)%8
    im=np.pad(img,((0,ph),(0,pw),(0,0)),mode='reflect'); mk=np.pad(mask,((0,ph),(0,pw)),mode='reflect')
    t=torch.from_numpy(im).permute(2,0,1)[None].float()/255.; m=torch.from_numpy((mk>127).astype(np.float32))[None,None]
    with torch.no_grad(): o=lama(t,m)
    return np.clip(o[0].permute(1,2,0).numpy()*255,0,255).astype(np.uint8)[:H,:W]
def extend(img, top, bot, work=640):
    """outpaint `top`/`bot` px; done coarse-to-fine: LaMa at low res for structure, then at full res for texture."""
    H,W=img.shape[:2]; NH=H+top+bot
    cv_=cv2.copyMakeBorder(img,top,bot,0,0,cv2.BORDER_REPLICATE)
    m=np.zeros((NH,W),np.uint8); m[:top]=255; m[NH-bot:]=255
    s=work/W; lo=cv2.resize(cv_,(work,round(NH*s)),interpolation=cv2.INTER_AREA); lm=cv2.resize(m,(work,round(NH*s)),interpolation=cv2.INTER_NEAREST)
    lo=run_lama(lo,lm)
    guess=cv2.resize(lo,(W,NH),interpolation=cv2.INTER_CUBIC); guess[top:top+H]=img
    full=run_lama(guess,m); full[top:top+H]=img
    # feather the seam a little
    out=full.copy()
    for y0,dirn in ((top,1),(top+H,-1)):
        for d in range(12):
            a=(d+1)/13; y=y0+d*dirn if dirn==1 else y0-1-d
            out[y]=(img[y-top]*a+full[y]*(1-a)).astype(np.uint8) if 0<=y-top<H else out[y]
    return out
if __name__=='__main__':
    cfg=json.load(open('work/vert_wide.json'))
    only=sys.argv[1:]
    for n,c in cfg.items():
        if only and n not in only: continue
        A=load(c['after']); B=load(c['before']); M=np.array(Image.open(c['mask']).convert('L'))
        if c.get('pre_scale'):
            f=c['pre_scale']; A=cv2.resize(A,None,fx=f,fy=f,interpolation=cv2.INTER_AREA); B=cv2.resize(B,None,fx=f,fy=f,interpolation=cv2.INTER_AREA); M=cv2.resize(M,None,fx=f,fy=f)
        H,W=A.shape[:2]; Wc=min(W,c['crop_w']); x0=int(np.clip(c['cx']-Wc//2,0,W-Wc)); sl=np.s_[:,x0:x0+Wc]
        A,B,M=A[sl],B[sl],M[sl]
        NH=round(Wc*16/9); ext=max(0,NH-H); top=round(ext*c.get('top_share',0.45)); bot=ext-top
        E=extend(B,top,bot)                      # extension built from the element-free image
        Aext=E.copy(); Aext[top:top+H]=A; Bext=E
        Mext=np.zeros(E.shape[:2],np.uint8); Mext[top:top+H]=M
        for k,v in (('after',Aext),('before',Bext)): Image.fromarray(v).save(f'work/vertw/{n}_{k}_raw.png')
        Image.fromarray(Mext).save(f'work/vertw/{n}_mask_raw.png'); print(n,'crop',x0,x0+Wc,'ext top',top,'bot',bot,flush=True)
