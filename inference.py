import os
import cv2
import time
import torch
import subprocess
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import seaborn as sns

from random import random
from tqdm import tqdm
from PIL import Image
from umap import UMAP
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import median_filter, gaussian_filter
from scipy.spatial.distance import pdist, squareform
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import PowerNorm
from sklearn.decomposition import PCA 

from visualize import predSmall

plt.style.use('dark_background')
plt.rcParams['animation.ffmpeg_path'] ='C:\\Program Files\\ImageMagick-7.1.0-Q16-HDRI\\ffmpeg.exe'


from matplotlib.animation import FFMpegWriter


class FasterFFMpegWriter(FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self._w, self._h)
            self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            self.fig.canvas.draw()
            self._proc.stdin.write(self.fig.canvas.tostring_argb())
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                      'Stdout: {1} StdError: {2}. It may help to re-run '
                      'with --verbose-debug.'.format(e, out, err)) 


def getPeriodicity(periodLength):
    periodicity = torch.nn.functional.threshold(periodLength, 2, 0)
    periodicity = -torch.nn.functional.threshold(-periodLength, -1, -1)
    return periodicity


def getCount(periodLength):
    frac = 1/periodLength
    frac = torch.nan_to_num(frac, 0, 0, 0)
    count = torch.sum(frac, dim = [1])
    return count


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_saliency_video(model, frames, img_size=112, seq_len=64, device='cuda', filter_over_time=True, periodicity_grad=False):
    Xlist = []
    for img in frames:
        preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        frameTensor = preprocess(img).unsqueeze(0)
        Xlist.append(frameTensor)

    if len(Xlist) < seq_len:
        for i in range(seq_len - len(Xlist)):
            Xlist.append(Xlist[-1])

    video_tensor = torch.cat(Xlist).unsqueeze(0)
    # we need to find the gradient with respect to the input image, so we need to call requires_grad_ on it
    video_tensor.requires_grad_(True)
    # Forward pass through the model
    model_output, periodicity_output = model(video_tensor.to(device))
    period_length = model_output[0]
    periodicity = periodicity_output[0]
    
    # Compute the gradient of the output with respect to the video tensor
    if periodicity_grad:
        periodicity.mean().backward()
    else:
        period_length.mean().backward()
    gradient = video_tensor.grad.data

    saliency_maps = []
    for i in range(seq_len):
        frame_gradient = gradient[0][i]
        saliency_maps.append(frame_gradient.cpu().numpy())
    saliency = np.abs(np.array(saliency_maps))
    if filter_over_time:
        saliency = gaussian_filter(saliency, sigma=2)
    return saliency


def eval_full_video(test_video, model, device, img_size=112, both_feet=False, seq_len=64, stride_length=32, stride_pad=3, subsample_frames=1, 
                    animate=False, progress_func=None, html_gen_func=None,
                    median_pred_filter=True, median_img_filter=True, epoch=0, out_dir='tmp_imgs', gamma=0.4, cmap='inferno'):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(test_video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / subsample_frames)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    all_frames = []
    frame_i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        if frame_i % subsample_frames == 0:
            frame = cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            all_frames.append(img)
        frame_i += 1
    cap.release()
    for _ in range(seq_len):  # pad full sequence
        all_frames.append(img)

    embs = np.zeros((length + seq_len, 512))
    embs_overlaps = np.zeros(((length + seq_len, 512)))
    period_lengths = np.zeros(length + seq_len)
    periodicities = np.zeros(length + seq_len)
    period_length_overlaps = np.zeros(length + seq_len)
    for i in range(0, length, stride_length):
        if html_gen_func is not None and progress_func is not None:
            progress_func(html_gen_func(i, length, "Predicting jump count..."))
        torch.cuda.empty_cache()
        model.eval()
        _, periodLengthj, periodicityj, sim, z = predSmall(all_frames[i:i + seq_len], model, device, img_size=img_size)
        period_lengths[i:i+seq_len] += periodLengthj.squeeze().cpu().numpy()
        periodicities[i:i+seq_len] += periodicityj.squeeze().cpu().numpy()
        period_length_overlaps[i:i+seq_len] += 1
        embs[i + stride_pad:i + seq_len - stride_pad] += z.squeeze().cpu().numpy()[stride_pad: -stride_pad]
        embs_overlaps[i + stride_pad:i + seq_len - stride_pad] += np.ones(512)
        del periodLengthj, periodicityj, sim, z
    
    numofReps = 0
    count = []
    periodLength = np.divide(period_lengths, period_length_overlaps, where=period_length_overlaps!=0)[stride_pad:length]
    periodicity = np.divide(periodicities, period_length_overlaps, where=period_length_overlaps!=0)[stride_pad:length]
    
    if median_pred_filter:
        periodicity = medfilt(periodicity, 5)
    periodicity = sigmoid(periodicity)
    periodicity_mask = np.int32(periodicity > 0.95)
    for i in range(len(periodLength)):
        if periodLength[i] < 2 or periodicity_mask[i] == 0:
            numofReps += 0
        else:
            numofReps += max(0, periodicity_mask[i]/(periodLength[i]))
        count.append(round(float(numofReps), 2))
    count_pred = count[-1]
    if both_feet:
        count_pred = count_pred / 2
        count = np.array(count) / 2
    score = np.mean(periodicity)

    embs = np.divide(embs, embs_overlaps, where=embs_overlaps!=0)[stride_pad:length]
    embs = np.nan_to_num(embs)
    pcs = PCA(2).fit_transform(embs)
    fig, axs = plt.subplots(4, 1, figsize=(12, 7))
    axs[0].plot(periodicity)
    axs[0].set_title('Periodicity')
    axs[0].set_ylim(0, 1.1)

    axs[1].plot(periodLength, label='pred')
    axs[1].legend()
    axs[1].set_title('Period Length')
    axs[1].set_ylim(0, 32)

    axs[2].plot(pcs[..., 0])
    axs[2].set_title('PC 1')

    axs[3].plot(pcs[..., 1])
    axs[3].set_title('PC 2')
    plt.tight_layout()

    #wandb.log({"full_video": fig, 'Epoch': epoch, 'video_id': video_id})
    fig.savefig(f'{out_dir}/period_{epoch}.png')
    plt.close()

    try:
        z_umap = UMAP(transform_seed=36, random_state=42).fit_transform(embs)
        df = pd.DataFrame.from_dict({'PC_1': pcs[..., 0], 'PC_2': pcs[..., 1],
                                    'UMAP_1': z_umap[..., 0], 'UMAP_2': z_umap[..., 1],
                                    'periodicity': periodicity,
                                    'period_length': periodLength,
                                    'count': count})
                                    
        fig, axs = plt.subplots(2, 3, figsize=(14, 10))
        palette = 'viridis'
        axs[0][0].set_title('Periodicity')
        sns.scatterplot(data=df, x='PC_1', y='PC_2', hue='periodicity', ax=axs[0][0], palette=palette, legend=False)
        axs[0][1].set_title('Period Length')
        sns.scatterplot(data=df, x='PC_1', y='PC_2', hue='period_length', ax=axs[0][1], palette=palette, legend=False)
        axs[0][2].set_title('Count')
        sns.scatterplot(data=df, x='PC_1', y='PC_2', hue='count', ax=axs[0][2], palette=palette, legend=False)
        sns.scatterplot(data=df, x='UMAP_1', y='UMAP_2', hue='periodicity', ax=axs[1][0], palette=palette, legend=False)
        sns.scatterplot(data=df, x='UMAP_1', y='UMAP_2', hue='period_length', ax=axs[1][1], palette=palette, legend=False)
        sns.scatterplot(data=df, x='UMAP_1', y='UMAP_2', hue='count', ax=axs[1][2], palette=palette, legend=False)
        fig.savefig(f'{out_dir}/embeddings_{epoch}.png')
        plt.close()
    except Exception as e:
        print(e)

    full_sim = pdist(embs, metric='euclidean')
    tssm = squareform(full_sim)
    if median_img_filter:
        tssm = median_filter(tssm, size=3)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(tssm, cmap=cmap, norm=PowerNorm(gamma=gamma))
    fig.savefig(f'{out_dir}/tssm_{epoch}.png')
    #wandb.log({"full_tssm": wandb.Image(fig), 'Epoch': epoch, 'video_id': video_id})
    plt.close()

    if animate:
        animate_frames = len(count)
        print(animate_frames)
        anim_subsample = max(1, int(fps / 24))
        fig, ax = plt.subplots(figsize = (3, 3))
        canvas_width, canvas_height = fig.canvas.get_width_height()
        print(canvas_width, canvas_height)
        img_ax = ax

        #saliency_maps = generate_saliency_video(model, all_frames[:animate_frames], device=device, img_size=img_size)
        imgs = []
        for img in all_frames:
            width, height = img.size
            if width > height:
                img = img.resize((int(width / (height / img_size)), img_size))
            else:
                img = img.resize((img_size, int(height / (width / img_size))))
            width, height = img.size
            h_center, w_center = height / 2, width / 2
            h_start, w_start = int(h_center - img_size / 2), int(w_center - img_size / 2)
            cropped = img.crop((w_start, h_start, w_start + img_size, h_start + img_size))
            imgs.append(cropped)

        alpha=1.0
        colormap=plt.cm.OrRd
        h, w, _ = np.shape(imgs[0])
        wedge_x = 34 / canvas_width * w
        wedge_y = 34 / canvas_height * h
        wedge_r = 30 / canvas_height * h
        txt_x = 34 / canvas_width * w
        txt_y = 36 / canvas_height * h
        otxt_size = 25 / canvas_height * h
        wedge1 = matplotlib.patches.Wedge(
            center=(wedge_x, wedge_y),
            r=wedge_r,
            theta1=0,
            theta2=0,
            color=colormap(1.),
            alpha=alpha)
        wedge2 = matplotlib.patches.Wedge(
            center=(wedge_x, wedge_y),
            r=wedge_r,
            theta1=0,
            theta2=0,
            color=colormap(0.5),
            alpha=alpha)

        im = img_ax.imshow(cropped)

        img_ax.add_patch(wedge1)
        img_ax.add_patch(wedge2)
        txt = img_ax.text(
            txt_x,
            txt_y,
            '0',
            size=otxt_size,
            ha='center',
            va='center',
            alpha=0.9,
            color='white',
        )

        def animate_fn(i):
            if anim_subsample:
                i *= anim_subsample
            cropped = imgs[i + stride_pad]
            current_count = count[i] * 2 if both_feet else count[i]
            if current_count % 2 == 0:
                wedge1.set_color(colormap(1.0))
                wedge2.set_color(colormap(0.5))
            else:
                wedge1.set_color(colormap(0.5))
                wedge2.set_color(colormap(1.0))
            txt.set_text(int(current_count))

            wedge1.set_theta1(-90)
            wedge1.set_theta2(-90 - 360 * (1 - current_count % 1.0))
            wedge2.set_theta1(-90 - 360 * (1 - current_count % 1.0))
            wedge2.set_theta2(-90)
            
            im.set_data(cropped)
            img_ax.set_title(f"Time: {i / fps:.1f}")
            img_ax.set_xticks([])
            img_ax.set_yticks([])
            img_ax.spines['top'].set_visible(False)
            img_ax.spines['right'].set_visible(False)
            img_ax.spines['bottom'].set_visible(False)
            img_ax.spines['left'].set_visible(False)

        anim_start_time = time.time()
        # Open an ffmpeg process
        outf = f'{out_dir}/anim_{epoch}.mp4'
        cmdstring = ('ffmpeg', 
                    '-y', '-r', f'{30 if anim_subsample != 1 else int(fps)}', # overwrite, 24fps
                    '-s', f'{canvas_width}x{canvas_height}',
                    '-pix_fmt', 'argb',
                    '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                    '-c:v', 'libx264', # https://trac.ffmpeg.org/wiki/Encode/H.264
                    outf) # output encoding
        p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

        # Draw frames and write to the pipe
        anim_length = int(animate_frames / anim_subsample) - 1
        count_msg = f"Predicted Count (both feet): {count_pred * 2 if both_feet else count_pred:.1f}"
        for frame in range(anim_length):
            if html_gen_func is not None and progress_func is not None:
                progress_func(html_gen_func(frame, anim_length, 
                                            "Generating visualization video...", 
                                            count_msg))
            # draw the frame
            animate_fn(frame)
            fig.canvas.draw()

            # extract the image as an ARGB string
            string = fig.canvas.tostring_argb()

            # write to pipe
            p.stdin.write(string)

        # Finish up
        p.communicate()
        print(f"Animation done in {time.time() - anim_start_time:.2f} seconds")
        cvrt_string = f"ffmpeg -i {out_dir}/anim_{epoch}.mp4 -i {test_video} -map 0:v -map 1:a -y -r 24 -s {canvas_width}x{canvas_height} -c:v libvpx {out_dir}/anim_{epoch}.webm"
        os.system(cvrt_string)
    return count_msg