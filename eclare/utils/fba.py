import numpy as np


def fba(imgs, p="infinity"):
    # images to Fourier domain
    vs_hat = [np.fft.rfftn(img) for img in imgs]

    if p == "infinity" or p == "inf":
        out_img = np.max(vs_hat, axis=0)
    else:
        p = float(p)
        # calculate fourier magnitude weights
        denominator = np.sum([np.abs(v_hat) ** p for v_hat in vs_hat], axis=0)
        ws = [np.abs(v_hat) ** p / denominator for v_hat in vs_hat]

        out_img = np.sum([w * v_hat for w, v_hat in zip(ws, vs_hat)], axis=0)

    # return to image domain
    out_img = np.fft.irfftn(out_img).astype(np.float32)

    return out_img
