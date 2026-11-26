"""Align user's edited 'before' images to the originals and composite only the element region."""
import numpy as np, cv2, json
from PIL import Image
def load(p): return np.array(Image.open(p).convert('RGB'))
jobs={
 's1_lounge':    dict(orig='src/s1_lounge.png', edit='src/edits/s1_lounge_edit.png', mask='work/masks/s1_lounge_sam.png', grow=14, reach=110, down=150),
 's2_meeting':   dict(orig='work/s2_meeting_after_x4.png', edit='src/edits/s2_meeting_edit.png', mask='work/masks/s2_meeting_sam_x4.png', grow=40, reach=160, down=0, extra_lines=True),
 's4_reception_front': dict(orig='src/x_reception_front.png', edit='src/edits/s4_reception_front_edit.png', mask='work/masks/s4_reception_front_sam.png', grow=14, reach=90, down=60),
 's5_stairs':    dict(orig='src/s5_stairs.png', edit='src/edits/s5_stairs_edit.png', mask='work/masks/s5_stairs_sam.png', grow=14, reach=70, down=0),
 's6_elevators': dict(orig='src/s6_elevators.png', edit='src/edits/s6_elevators_edit.png', mask='work/masks/s6_elevators_sam.png', grow=10, reach=40, down=0),
}
sift=cv2.SIFT_create(6000)
for name,j in jobs.items():
    o=load(j['orig']); e=load(j['edit']); H,W=o.shape[:2]
    e=cv2.resize(e,(W,H),interpolation=cv2.INTER_CUBIC)
    m=(np.array(Image.open(j['mask']).convert('L'))>127).astype(np.uint8)*255
    if j.get('extra_lines'):   # meeting: cables + canopy (x4 coords)
        for x0,y0,x1,y1 in [(356,17,313,103),(380,17,422,103),(367,17,367,100)]: cv2.line(m,(x0*4,y0*4),(x1*4,y1*4),255,16)
        m[0:76,1396:1544]=255
    # --- register edit -> orig using features away from the element
    s=min(1.0,1600/W); os_=cv2.resize(o,None,fx=s,fy=s); es=cv2.resize(e,None,fx=s,fy=s)
    keep=cv2.resize(255-cv2.dilate(m,np.ones((61,61),np.uint8)),None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST)
    k1,d1=sift.detectAndCompute(cv2.cvtColor(es,cv2.COLOR_RGB2GRAY),keep); k2,d2=sift.detectAndCompute(cv2.cvtColor(os_,cv2.COLOR_RGB2GRAY),keep)
    mt=cv2.BFMatcher().knnMatch(d1,d2,k=2); good=[a for a,b in mt if a.distance<0.7*b.distance]
    p1=np.float32([k1[g.queryIdx].pt for g in good])/s; p2=np.float32([k2[g.trainIdx].pt for g in good])/s
    A,inl=cv2.estimateAffinePartial2D(p1,p2,method=cv2.RANSAC,ransacReprojThreshold=2.0,maxIters=5000)
    ew=cv2.warpAffine(e,A,(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
    res=np.linalg.norm((p1@A[:,:2].T+A[:,2])-p2,axis=1)[inl[:,0]>0]
    print(f'{name}: matches {len(good)} inliers {int(inl.sum())} scale {np.hypot(*A[0,:2]):.4f} shift ({A[0,2]:.1f},{A[1,2]:.1f}) err {np.median(res):.2f}px')
    # --- composite region = grown element mask  U  (changed pixels near the element)
    k=lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
    near=cv2.dilate(m,k(j['reach']))
    if j['down']: near=np.maximum(near,cv2.dilate(m,np.ones((j['down'],3),np.uint8),anchor=(1,0)))
    lo=cv2.cvtColor(cv2.GaussianBlur(o,(0,0),3),cv2.COLOR_RGB2LAB).astype(np.float32); le=cv2.cvtColor(cv2.GaussianBlur(ew,(0,0),3),cv2.COLOR_RGB2LAB).astype(np.float32)
    diff=(np.linalg.norm(lo-le,axis=2)>14).astype(np.uint8)*255
    diff=cv2.morphologyEx(diff,cv2.MORPH_OPEN,k(3)); diff=cv2.morphologyEx(diff,cv2.MORPH_CLOSE,k(9))
    comp=np.maximum(cv2.dilate(m,k(j['grow'])), diff&near)
    # keep only diff blobs connected to the element
    n,lab=cv2.connectedComponents(comp); ids=set(np.unique(lab[m>0]))-{0}
    comp=np.isin(lab,list(ids)).astype(np.uint8)*255
    # --- colour-match the edit to the original on a ring around the region
    ring=(cv2.dilate(comp,k(25))>0)&(comp==0)
    ec=ew.astype(np.float32).copy()
    for c in range(3):
        a=ew[...,c][ring].astype(np.float32); b=o[...,c][ring].astype(np.float32)
        g=np.clip(b.std()/(a.std()+1e-3),0.85,1.15); ec[...,c]=(ec[...,c]-a.mean())*g+b.mean()
    soft=cv2.GaussianBlur(comp.astype(np.float32)/255.,(0,0),4)[...,None]
    out=(o*(1-soft)+ec*soft).clip(0,255).astype(np.uint8)
    dst=f'work/{name}_before_x4.png' if name=='s2_meeting' else f'work/{name}_before.png'
    Image.fromarray(out).save(dst); Image.fromarray(comp).save(f'work/masks/{name}_comp.png'); print('  ->',dst, 'region %.1f%%'%(100*(comp>0).mean()))
