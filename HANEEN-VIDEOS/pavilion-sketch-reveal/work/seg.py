import sys, json, numpy as np, torch
from PIL import Image
from transformers import SamModel, SamProcessor
torch.set_num_threads(4)
M="facebook/sam-vit-large"
proc=SamProcessor.from_pretrained(M); model=SamModel.from_pretrained(M).eval()
cfg=json.load(open('work/boxes.json'))
only=sys.argv[1:]
for name,c in cfg.items():
    if only and name not in only: continue
    im=Image.open(c['src']).convert('RGB'); W,H=im.size
    union=np.zeros((H,W),bool)
    boxes=c['boxes']
    inp=proc(im, input_boxes=[boxes], return_tensors='pt')
    with torch.no_grad(): out=model(**inp, multimask_output=True)
    masks=proc.image_processor.post_process_masks(out.pred_masks, inp['original_sizes'], inp['reshaped_input_sizes'])[0]  # (nbox,3,H,W)
    scores=out.iou_scores[0]  # (nbox,3)
    for i,b in enumerate(boxes):
        # choose mask: prefer the one with best score but covering a reasonable portion of the box
        x0,y0,x1,y1=b; area=(x1-x0)*(y1-y0)
        best=None
        for k in range(3):
            m=masks[i,k].numpy(); inbox=m[y0:y1,x0:x1].sum()/area; out_=m.sum()-m[y0:y1,x0:x1].sum()
            s=float(scores[i,k]) - (0.5 if inbox<c.get('min_fill',0.25) else 0) - out_/area
            if best is None or s>best[0]: best=(s,k,m,inbox)
        print(name,i,b,'k',best[1],'fill %.2f'%best[3],'score %.3f'%best[0])
        union|=best[2]
    Image.fromarray((union*255).astype(np.uint8)).save(f'work/masks/{name}_sam.png')
