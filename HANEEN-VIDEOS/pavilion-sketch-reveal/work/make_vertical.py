"""Crop each scene to a full-bleed 9:16 window around its element and upscale (Real-ESRGAN x2) to 1080x1920."""
import torch, numpy as np, cv2, json
from PIL import Image
from spandrel import ModelLoader
torch.set_num_threads(4)
up=ModelLoader().load_from_file('/tmp/claude-0/-home-user-357-profile/6bba53e8-0a57-5a08-91ac-769c623010ed/scratchpad/models/RealESRGAN_x2plus.pth').eval()
def load(p): return np.array(Image.open(p).convert('RGB'))
def esr(a):
    t=torch.from_numpy(a.astype(np.float32)/255.).permute(2,0,1)[None]
    with torch.no_grad(): o=up(t)
    return (o[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
S={ # name: (after, before, mask, centre_x in source px, already_hires)
 's1_lounge':  ('src/s1_lounge.png','work/s1_lounge_before.png','work/masks/s1_lounge_sam.png',404,False),
 's2_meeting': ('work/s2_meeting_after_x4.png','work/s2_meeting_before_x4.png','work/masks/s2_meeting_sam_x4.png',1468,True),
 's3_painting':('src/s3_painting.png','work/s3_painting_before.png','work/masks/s3_painting_sam.png',823,False),
 's4_reception_front':('src/x_reception_front.png','work/s4_reception_front_before.png','work/masks/s4_reception_front_sam.png',836,False),
 's5_stairs':  ('src/s5_stairs.png','work/s5_stairs_before.png','work/masks/s5_stairs_sam.png',745,False),
 's6_elevators':('src/s6_elevators.png','work/s6_elevators_before.png','work/masks/s6_elevators_sam.png',848,False),
}
for n,(a,b,m,cx,hi) in S.items():
    A=load(a); B=load(b); M=np.array(Image.open(m).convert('L')); H,W=A.shape[:2]
    w=round(H*9/16); x0=int(np.clip(cx-w//2,0,W-w)); sl=np.s_[:,x0:x0+w]
    A,B,M=A[sl],B[sl],M[sl]
    if not hi: A,B=esr(A),esr(B)
    A=cv2.resize(A,(1080,1920),interpolation=cv2.INTER_AREA); B=cv2.resize(B,(1080,1920),interpolation=cv2.INTER_AREA)
    M=cv2.resize(M,(1080,1920),interpolation=cv2.INTER_LINEAR)
    Image.fromarray(A).save(f'work/vert/{n}_after.png'); Image.fromarray(B).save(f'work/vert/{n}_before.png'); Image.fromarray(M).save(f'work/vert/{n}_mask.png')
    print(n,'crop x',x0,x0+w,flush=True)
