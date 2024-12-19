import time

import numpy as np
from matplotlib import cycler

try:
    import matplotlib.pyplot as plt
    from IPython import display
    from PIL import Image
except ImportError:
    raise ImportError('Plotting extras must be installed to use plotting featues. '
                      'Install using eclare[plot].')
    
def set_display_mode(mode='dark', figsize=(16, 9), fontsize=18):
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['image.cmap'] = 'Greys_r'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['image.interpolation'] = 'nearest'
    
    if mode == 'dark':
        bg_color = 'black'
        fg_color = 'white'
    else:
        bg_color = 'white'
        fg_color = 'black'
        
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['axes.facecolor'] = bg_color
    plt.rcParams['axes.edgecolor'] = fg_color
    plt.rcParams['axes.prop_cycle'] = cycler(color=[fg_color]) 
    plt.rcParams['text.color'] = fg_color
    plt.rcParams['axes.labelcolor'] = fg_color
    plt.rcParams['xtick.color'] = fg_color
    plt.rcParams['ytick.color'] = fg_color

def multiplot(imgs, titles, vmins=None, vmaxs=None, figsize=(16, 9), target_shape=None, out_fpath=None, disp=True):
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(imgs),
        figsize=figsize
    )
    
    if vmins is None:
        vmins = [img.min() for img in imgs]
    if vmaxs is None:
        vmaxs = [img.max() for img in imgs]

    for ax, img, title, vmin, vmax in zip(axs, imgs, titles, vmins, vmaxs):
        if vmin is None:
            vmin = img.min()
        if vmax is None:
            vmax = img.max()
            
        if target_shape is not None:
            # check slice shapes
            if img.shape != (target_shape[0], target_shape[1]):
                # PIL resize needs (y, x)
                img = np.array(Image.fromarray(img)
                             .resize((target_shape[1], target_shape[0]),
                                     Image.NEAREST))
        ax.imshow(np.rot90(img), vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    if out_fpath is not None:
        plt.savefig(out_fpath)
    if not disp:
        plt.close()
    else:
        plt.show()


def center_vol_plot(img_vol, target_shape=None, vmin=None, vmax=None, figsize=(16, 9)):
    cs = [c // 2 for c in img_vol.shape]

    x = img_vol[cs[0], :, :]
    y = img_vol[:, cs[1], :]
    z = img_vol[:, :, cs[2]]
    
    if vmin is None:
        vmin = img_vol.min()
    if vmax is None:
        vmax = img_vol.max()

    # For anisotropic images, provide `target_shape` for NN interp
    if target_shape is not None:
        # check slice shapes
        if x.shape != (target_shape[1], target_shape[2]):
            # PIL resize needs (y, x)
            x = np.array(Image.fromarray(x)
                         .resize((target_shape[2], target_shape[1]),
                                 Image.NEAREST))
        if y.shape != (target_shape[0], target_shape[2]):
            y = np.array(Image.fromarray(y)
                         .resize((target_shape[2], target_shape[0]),
                                 Image.NEAREST))
        if z.shape != (target_shape[0], target_shape[1]):
            z = np.array(Image.fromarray(z)
                         .resize((target_shape[1], target_shape[0]),
                                 Image.NEAREST))

    multiplot([x, y, z], [1, 2, 3], 
              vmins=[vmin, vmin, vmin], 
              vmaxs=[vmax, vmax, vmax],
              figsize=figsize,
             )


def anim_paired_patches(lr_patch, hr_patch):
    fig, axs = plt.subplots(1, 2)

    vmin = lr_patch.min()
    vmax = lr_patch.max()

    axs[0].imshow(np.rot90(lr_patch), vmin=vmin, vmax=vmax)
    axs[1].imshow(np.rot90(hr_patch), vmin=vmin, vmax=vmax)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    display.display(plt.show())
    display.clear_output(wait=True)
    time.sleep(.1)
