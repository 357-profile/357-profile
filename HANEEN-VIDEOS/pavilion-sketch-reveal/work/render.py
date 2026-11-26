"""Sketch -> boom reveal renderer. Usage: python3 work/render.py work/scenes.json out.mp4 [--preview NAME]"""
import sys, json, math, subprocess, numpy as np, cv2
from PIL import Image
OW,OH,FPS=1920,1080,30
rng=np.random.default_rng(357)

def load(p): return np.array(Image.open(p).convert('RGB'))
def fit(img, crop, interp=cv2.INTER_LANCZOS4):
    """cover-fit to OWxOH; crop=[x0,y0,x1,y1] in source px (optional)"""
    if crop: x0,y0,x1,y1=crop; img=img[y0:y1,x0:x1]
    h,w=img.shape[:2]; s=max(OW/w,OH/h); nw,nh=round(w*s),round(h*s)
    r=cv2.resize(img,(nw,nh),interpolation=interp); ox,oy=(nw-OW)//2,(nh-OH)//2
    return r[oy:oy+OH,ox:ox+OW]
def smoothstep(x): x=np.clip(x,0,1); return x*x*(3-2*x)

def resample(pts, step):
    pts=np.asarray(pts,np.float32); d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
    if d[-1]<1: return pts
    n=max(2,int(d[-1]/step)); t=np.linspace(0,d[-1],n)
    return np.c_[np.interp(t,d,pts[:,0]),np.interp(t,d,pts[:,1])]
def smooth_closed(pts, sig):
    k=int(3*sig)|1; pad=np.r_[pts[-k:],pts,pts[:k]]
    g=cv2.GaussianBlur(pad.reshape(-1,1,2).astype(np.float32),(1,2*k+1),sig).reshape(-1,2)
    return g[k:-k]
def wobble(n, amp, freq):
    t=np.arange(n); ph=rng.uniform(0,6.28,3)
    return amp*(np.sin(t*freq+ph[0])*.6+np.sin(t*freq*2.3+ph[1])*.3+np.sin(t*freq*5.1+ph[2])*.1)

def colorize(after, sat=1.25, val=1.0, blur=5):
    b=cv2.GaussianBlur(after,(0,0),blur); hsv=cv2.cvtColor(b,cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1]=np.clip(hsv[...,1]*sat,0,255); hsv[...,2]=np.clip(hsv[...,2]*val,0,255)
    return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB)

class Stroke:
    def __init__(s, pts, width, colors, t0, t1, alpha=0.95):
        s.p=np.asarray(pts,np.float32); s.w=width; s.c=colors; s.t0=t0; s.t1=t1; s.drawn=1; s.alpha=alpha
        s.n=len(s.p)

def build_scene(sc):
    after=fit(load(sc['after']),sc.get('crop')); before=fit(load(sc['before']),sc.get('crop'))
    if sc.get('before_dim'): before=(before.astype(np.float32)*sc['before_dim']).astype(np.uint8)
    m=fit(np.array(Image.open(sc['mask']).convert('L')),sc.get('crop'),cv2.INTER_LINEAR)
    m=(m>127).astype(np.uint8)*255
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    T=sc['timing']; W=sc.get('width',15)
    pal=colorize(after, sc.get('sat',1.25), sc.get('val',1.0))
    # dominant colour -> outline colour
    px=after[m>0].reshape(-1,3).astype(np.float32); med=np.median(px,0)
    hsv=cv2.cvtColor(np.uint8([[med]]),cv2.COLOR_RGB2HSV)[0,0].astype(float)
    hsv[1]=min(255,hsv[1]*1.7+45) if hsv[1]>28 else hsv[1]; hsv[2]=float(np.clip(hsv[2]*sc.get('outline_val',0.6),55,175))
    oc=tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]),cv2.COLOR_HSV2RGB)[0,0])
    if sc.get('outline_rgb'): oc=tuple(sc['outline_rgb'])
    strokes_fill=[]; strokes_line=[]
    n,lab,st,cen=cv2.connectedComponentsWithStats(m)
    comps=[i for i in range(1,n) if st[i,4]>sc.get('min_area',1500)]
    comps.sort(key=lambda i: st[i,0])
    nC=len(comps)
    # ---- outlines
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnts=[c for c in cnts if cv2.contourArea(c)>sc.get('min_area',1500)]
    cnts.sort(key=lambda c: c[:,0,0].min())
    for j,c in enumerate(cnts):
        p=resample(np.r_[c[:,0,:],c[:1,0,:]],3); p=smooth_closed(p,4)
        nrm=np.gradient(p,axis=0); nrm=np.c_[-nrm[:,1],nrm[:,0]]; nrm/=np.linalg.norm(nrm,axis=1,keepdims=True)+1e-6
        p=p+nrm*wobble(len(p),2.5,0.035)[:,None]
        # start at top-left-ish point, overshoot 4%
        k=int(np.argmin(p[:,0]+p[:,1])); p=np.r_[p[k:],p[:k]]; p=np.r_[p,p[:max(3,len(p)//25)]]
        stg=j/max(1,len(cnts))*T['line_stagger']
        strokes_line.append(Stroke(p,sc.get('outline_w',6),None,T['line0']+stg,T['line1']+stg*0.3,0.97))
    # ---- inner detail lines (from real edges of the element)
    if sc.get('details',True):
        g=cv2.GaussianBlur(cv2.cvtColor(after,cv2.COLOR_RGB2GRAY),(0,0),sc.get('det_blur',2.2))
        e=cv2.Canny(g,sc.get('canny',[35,90])[0],sc.get('canny',[35,90])[1])
        er=cv2.erode(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))); e[er==0]=0
        dc,_=cv2.findContours(e,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        dc=[d for d in dc if cv2.arcLength(d,False)>sc.get('det_min',90)]
        dc.sort(key=lambda d:-cv2.arcLength(d,False)); dc=dc[:sc.get('det_max',40)]
        dc.sort(key=lambda d: d[:,0,0].min())
        for j,d in enumerate(dc):
            p=d[:,0,:].astype(np.float32)
            # contours from Canny are traced twice (there & back) -> keep first half
            p=p[:max(2,len(p)//2+1)]
            p=resample(p,3)
            if len(p)>9: p=cv2.GaussianBlur(p.reshape(-1,1,2),(1,9),2).reshape(-1,2)
            stg=j/max(1,len(dc))*T['line_stagger']
            strokes_line.append(Stroke(p,sc.get('det_w',3.5),None,T['line0']+0.15+stg,T['line1']+stg*0.4,0.85))
    # ---- zigzag scribble per component
    for ci,i in enumerate(comps):
        cm=(lab==i).astype(np.uint8); x,y,w,h,a=st[i]
        vertical_prog = h>sc.get('vert_ratio',1.35)*w
        s=W*sc.get('density',0.62)
        pts=[]
        if not vertical_prog:
            cols=np.arange(x+2,x+w-2,s)
            for k,xx in enumerate(cols):
                col=np.where(cm[:,int(xx)])[0]
                if len(col)==0: continue
                top,bot=col.min(),col.max(); span=bot-top
                yy=(top-rng.uniform(0,.05)*span) if k%2==0 else (bot+rng.uniform(0,.05)*span)
                pts.append((xx+rng.normal(0,s*0.18),yy))
        else:
            rows=np.arange(y+2,y+h-2,s)
            for k,yy in enumerate(rows):
                row=np.where(cm[int(yy),:])[0]
                if len(row)==0: continue
                l,r=row.min(),row.max(); span=r-l
                xx=(l-rng.uniform(0,.05)*span) if k%2==0 else (r+rng.uniform(0,.05)*span)
                pts.append((xx,yy+rng.normal(0,s*0.18)))
        if len(pts)<2: continue
        bowed=[pts[0]]
        for (ax,ay),(bx,by) in zip(pts[:-1],pts[1:]):
            dx,dy=bx-ax,by-ay; L=math.hypot(dx,dy)+1e-6; b=rng.normal(0,0.035)*L
            for t in (0.25,0.5,0.75):
                o=b*4*t*(1-t); bowed.append((ax+dx*t-dy/L*o, ay+dy*t+dx/L*o))
            bowed.append((bx,by))
        p=resample(bowed,5)
        cols_=pal[np.clip(p[:,1].astype(int),0,OH-1),np.clip(p[:,0].astype(int),0,OW-1)]
        stg=ci/max(1,nC)*T['fill_stagger']
        strokes_fill.append(Stroke(p,W,cols_,T['fill0']+stg,T['fill1']-(T['fill_stagger']-stg)*0.5,0.95))
    clip=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))).astype(np.float32)/255.
    clip=cv2.GaussianBlur(clip,(0,0),1.5)
    cx,cy=sc.get('zoom_center',[OW/2,OH/2])
    return dict(after=after,before=before,mask=m,clip=clip,fill=strokes_fill,line=strokes_line,oc=oc,T=T,center=(cx,cy),sc=sc)

def draw_progress(layer, a_layer, stroke, f, color=None):
    """incrementally draw stroke up to its progress at frame f"""
    if f<stroke.t0: return
    pr=smoothstep((f-stroke.t0)/max(1,(stroke.t1-stroke.t0))) if sc_ease else min(1,(f-stroke.t0)/(stroke.t1-stroke.t0))
    upto=int(round(pr*(stroke.n-1)))+1
    if upto<=stroke.drawn: return
    w=int(round(stroke.w))
    for k in range(stroke.drawn-1, upto-1):
        a=tuple(int(round(v*4)) for v in stroke.p[k]); b=tuple(int(round(v*4)) for v in stroke.p[k+1])
        c=color if color is not None else tuple(int(v) for v in stroke.c[k])
        cv2.line(layer,a,b,c,w,cv2.LINE_AA,shift=2); cv2.line(a_layer,a,b,stroke.alpha,w,cv2.LINE_AA,shift=2)
    stroke.drawn=upto
sc_ease=True

def camera(img, z, cx, cy):
    if abs(z-1)<1e-4: return img
    w,h=OW/z,OH/z; x0=np.clip(cx-w/2,0,OW-w); y0=np.clip(cy-h/2,0,OH-h)
    M=np.float32([[z,0,-x0*z],[0,z,-y0*z]])
    return cv2.warpAffine(img,M,(OW,OH),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)

def render_scene(S, emit):
    T=S['T']; nF=T['frames']; boom=T['boom']
    fill_rgb=np.zeros((OH,OW,3),np.uint8); fill_a=np.zeros((OH,OW),np.float32)
    line_rgb=np.zeros((OH,OW,3),np.uint8); line_a=np.zeros((OH,OW),np.float32)
    before=S['before'].astype(np.float32); after=S['after'].astype(np.float32)
    for f in range(nF):
        tz=1.0+T.get('zoom',0.035)*f/nF
        if f<boom:
            for s in S['fill']: draw_progress(fill_rgb,fill_a,s,f)
            for s in S['line']: draw_progress(line_rgb,line_a,s,f,S['oc'])
            fa=(fill_a*S['clip'])[...,None]; la=line_a[...,None]
            fr=before*(1-fa)+fill_rgb*fa; fr=fr*(1-la)+line_rgb*la
        else:
            k=f-boom; fr=after.copy()
            tz+=T.get('punch',0.06)*math.exp(-k/2.6)
            fl=T.get('flash',0.6)*math.exp(-k/1.6)
            if fl>0.01: fr=fr*(1-fl)+np.array([255,246,228],np.float32)*fl
        emit(camera(np.clip(fr,0,255).astype(np.uint8),tz,*S['center']))

if __name__=='__main__':
    cfg=json.load(open(sys.argv[1])); out=sys.argv[2]
    prev=sys.argv[sys.argv.index('--preview')+1] if '--preview' in sys.argv else None
    if prev:
        import os; os.makedirs('work/preview',exist_ok=True)
        sc=[s for s in cfg['scenes'] if s['name']==prev][0]; sc['timing']={**cfg['timing'],**sc.get('timing',{})}
        S=build_scene(sc); frames=[]
        render_scene(S, frames.append)
        T=S['T']
        picks=[0,T['line0']+8,T['fill0']+8,(T['fill0']+T['fill1'])//2,T['boom']-1,T['boom']+1,T['frames']-1]
        tiles=[cv2.resize(frames[i],(640,360),interpolation=cv2.INTER_AREA) for i in picks]+[np.zeros((360,640,3),np.uint8)]
        sheet=np.vstack([np.hstack(tiles[0:4]),np.hstack(tiles[4:8])])
        Image.fromarray(sheet).save(f'work/preview/{prev}.jpg',quality=88)
        Image.fromarray(frames[T['boom']-1]).save(f'work/preview/{prev}_scribble.jpg',quality=90)
        print('preview ok',len(S['fill']),'fill strokes',len(S['line']),'lines'); sys.exit()
    ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}','-r',str(FPS),'-i','-',
        '-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-shortest','-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','128k','-movflags','+faststart',out],stdin=subprocess.PIPE)
    for sc in cfg['scenes']:
        sc['timing']={**cfg['timing'],**sc.get('timing',{})}
        S=build_scene(sc); render_scene(S, lambda fr: ff.stdin.write(fr.tobytes())); print('scene',sc['name'],flush=True)
    ff.stdin.close(); ff.wait(); print('wrote',out)
