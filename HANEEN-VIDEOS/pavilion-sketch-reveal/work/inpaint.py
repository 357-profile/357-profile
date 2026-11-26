import sys, json, numpy as np, torch, cv2
from PIL import Image
torch.set_num_threads(4)
MD='/tmp/claude-0/-home-user-357-profile/6bba53e8-0a57-5a08-91ac-769c623010ed/scratchpad/models/big-lama.pt'
lama=torch.jit.load(MD, map_location='cpu').eval()
cfg=json.load(open('work/boxes.json'))
def run_lama(img, mask):
    H,W=mask.shape; ph=(8-H%8)%8; pw=(8-W%8)%8
    im=np.pad(img,((0,ph),(0,pw),(0,0)),mode='reflect'); mk=np.pad(mask,((0,ph),(0,pw)),mode='reflect')
    t=torch.from_numpy(im).permute(2,0,1)[None].float()/255.; m=torch.from_numpy((mk>127).astype(np.float32))[None,None]
    with torch.no_grad(): o=lama(t,m)
    return np.clip(o[0].permute(1,2,0).numpy()*255,0,255).astype(np.uint8)[:H,:W]
c_pad=1.0
def inpaint_hd(img, mask, maxside):
    """Crop around mask, resize so the crop's long side <= maxside, inpaint, resize back, paste."""
    ys,xs=np.where(mask>127); H,W=mask.shape
    x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max(); pad=int(0.6*max(x1-x0,y1-y0))+32
    pad=int(pad*c_pad)
    x0,y0=max(0,x0-pad),max(0,y0-pad); x1,y1=min(W,x1+pad),min(H,y1+pad)
    ci=img[y0:y1,x0:x1]; cm=mask[y0:y1,x0:x1]; h,w=cm.shape; s=min(1.0,maxside/max(h,w))
    if s<1:
        ri=cv2.resize(ci,(int(w*s),int(h*s)),interpolation=cv2.INTER_AREA); rm=cv2.resize(cm,(int(w*s),int(h*s)),interpolation=cv2.INTER_NEAREST)
        ro=run_lama(ri,rm); ro=cv2.resize(ro,(w,h),interpolation=cv2.INTER_CUBIC)
    else: ro=run_lama(ci,cm)
    out=img.copy(); out[y0:y1,x0:x1]=ro; return out
for name,c in cfg.items():
    if len(sys.argv)>1 and name not in sys.argv[1:]: continue
    img=np.array(Image.open(c['src']).convert('RGB'))
    m=np.array(Image.open(f'work/masks/{name}_sam.png'))
    if 'extra' in c:  # extra rectangles to remove (rods, small parts)
        for x0,y0,x1,y1 in c['extra']: m[y0:y1,x0:x1]=255
    for x0,y0,x1,y1,w in c.get('lines',[]): cv2.line(m,(x0,y0),(x1,y1),255,w)
    d=c.get('dilate',12)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*d+1,2*d+1)); md=cv2.dilate(m,k)
    if c.get('shadow'):   # extend downward to catch contact shadows
        sh=c['shadow']; md=np.maximum(md, cv2.dilate(m, np.ones((sh,3),np.uint8), anchor=(1,0)))
    Image.fromarray(md).save(f'work/masks/{name}_inpaint.png')
    if c.get('split'):
        n,lab=cv2.connectedComponents(md); out=img.copy()
        for i in range(1,n):
            mi=((lab==i)*255).astype(np.uint8)
            for _ in range(c.get('passes',1)): out=inpaint_hd(out, mi, c.get('maxside',1024))
    else:
        out=img.copy()
        for _ in range(c.get('passes',1)): out=inpaint_hd(out, md, c.get('maxside',1024))
    soft=cv2.GaussianBlur(md.astype(np.float32)/255.,(0,0),3)[...,None]
    res=(img*(1-soft)+out*soft).astype(np.uint8)
    Image.fromarray(res).save(f'work/{name}_before.png'); print('done',name,flush=True)
