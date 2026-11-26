import torch, numpy as np, cv2
from PIL import Image
from spandrel import ModelLoader
torch.set_num_threads(4)
m=ModelLoader().load_from_file('/tmp/claude-0/-home-user-357-profile/6bba53e8-0a57-5a08-91ac-769c623010ed/scratchpad/models/RealESRGAN_x4plus.pth').eval()
for src,dst in [('src/s2_meeting.jpg','work/s2_meeting_after_x4.png'),('work/s2_meeting_before.png','work/s2_meeting_before_x4.png')]:
    a=np.array(Image.open(src).convert('RGB')).astype(np.float32)/255.
    t=torch.from_numpy(a).permute(2,0,1)[None]
    with torch.no_grad(): o=m(t)
    o=(o[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
    Image.fromarray(o).save(dst); print(dst,o.shape,flush=True)
mk=np.array(Image.open('work/masks/s2_meeting_sam.png')); 
Image.fromarray(cv2.resize(mk,(mk.shape[1]*4,mk.shape[0]*4),interpolation=cv2.INTER_LINEAR)).save('work/masks/s2_meeting_sam_x4.png')
