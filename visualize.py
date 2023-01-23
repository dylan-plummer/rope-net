import torch
import cv2
from PIL import Image
from torchvision import transforms


def pred_rep(vidPath, model, device, numdivs = 1, img_size=112):
    countbest=[]
    periodicitybest = []
    Xbest = None
    countbest = [-1]
    simsbest = []
    periodbest = []
    
    for i in range(numdivs, numdivs+1):
        frames = getFrames(vidPath, 64*i)
        periodicity = []
        periodLength = []
        sims = []
        X = []
        embs = []
        for j in range(i):
            x, periodLengthj, periodicityj, sim, z = predSmall(frames[j*64:(j+1)*64], model, device, img_size=img_size)
            periodicity.extend(list(periodicityj.squeeze().cpu().numpy()))
            periodLength.extend(list(periodLengthj.squeeze().cpu().numpy()))
            X.append(x)
            sims.append(sim)
            embs.append(z)
        
        X = torch.cat(X)
        numofReps = 0
        count = []
        for i in range(len(periodLength)):
            if periodLength[i] == 0:
                numofReps += 0
            else:
                numofReps += max(0, periodicity[i]/(periodLength[i]))

            count.append(float(numofReps))
        
        if count[-1] > countbest[-1]:
            countbest = count
            Xbest = X
            periodicitybest = periodicity
            simsbest = sims
            periodbest = periodLength
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    return inv_normalize(Xbest), countbest, periodicitybest, periodbest, simsbest

                
def getFrames(vidPath, num_frames=64):
    frames = []
    cap = cv2.VideoCapture(vidPath)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        img = Image.fromarray(frame)
        frames.append(img)
    cap.release()
    
    newFrames = []
    for i in range(1, num_frames + 1):
        newFrames.append(frames[i * len(frames)//num_frames  - 1])
    
    return newFrames


def pred_overlapping_batch(frames_list, model, device, img_size=112, seq_len=64, center_crop=True):
    batch = []
    for frames in frames_list:
        Xlist = []
        #print(frames)
        #print(frames[0])
        h = frames[0].size[0]
        w = frames[0].size[1]
        if center_crop:
            if h > w:
                center_crop_h = w 
                center_crop_w = w
            else:
                center_crop_h = h
                center_crop_w = h
        else:
            center_crop_h = h
            center_crop_w = w
        for img in frames:
            transforms_list = []
            if center_crop:
                transforms_list.append(transforms.CenterCrop((center_crop_h, center_crop_w)))

            transforms_list += [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            preprocess = transforms.Compose(transforms_list)
            frameTensor = preprocess(img).unsqueeze(0)
            Xlist.append(frameTensor)

        if len(Xlist) < seq_len:
            for i in range(seq_len - len(Xlist)):
                Xlist.append(Xlist[-1])
        X = torch.cat(Xlist)
        batch.append(X.unsqueeze(0))
    batch = torch.cat(batch)
    with torch.no_grad():
        model.eval()
        y1pred, y2pred, sim, z = model(batch.to(device).float(), True, True)
    periodLength = y1pred
    periodicity = y2pred
    
    sim = sim.detach().cpu().numpy()
    
    return batch, periodLength, periodicity, sim, z
    


def predSmall(frames, model, device, img_size=112, seq_len=64, center_crop=True):
    Xlist = []
    h = frames[0].size[0]
    w = frames[0].size[1]
    if center_crop:
        if h > w:
            center_crop_h = w 
            center_crop_w = w
        else:
            center_crop_h = h
            center_crop_w = h
    else:
        center_crop_h = h
        center_crop_w = w
    for img in frames:
        transforms_list = []
        if center_crop:
            transforms_list.append(transforms.CenterCrop((center_crop_h, center_crop_w)))

        transforms_list += [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        preprocess = transforms.Compose(transforms_list)
        frameTensor = preprocess(img).unsqueeze(0)
        Xlist.append(frameTensor)

    if len(Xlist) < seq_len:
        for i in range(seq_len - len(Xlist)):
            Xlist.append(Xlist[-1])

    X = torch.cat(Xlist)
    with torch.no_grad():
        model.eval()
        y1pred, y2pred, sim, z = model(X.unsqueeze(0).to(device), True, True)
    
    periodLength = y1pred
    periodicity = y2pred
    #print(periodLength.squeeze())
    #print(periodicity.squeeze())
    
    sim = sim[0,:,:,:]
    sim = sim.detach().cpu().numpy()
    
    return X, periodLength, periodicity, sim, z