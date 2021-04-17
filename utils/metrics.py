import numpy as np
import torch

from piqa import SSIM, HaarPSI, PSNR, MS_SSIM, MS_GMSD, MDSI


def compute_metrics(real, fakes, size, out):

    p, s, h, ms, md = [], [], [], [], []

    ssim = SSIM().cpu()
    psnr = PSNR()
    haar = HaarPSI()
    msssim = MS_SSIM()
    ms_gmsd = MS_GMSD()
    mdsi = MDSI()

    if len(real[0]) > len(fakes[-1]):
        thres = len(fakes[-1])

    elif len(real[0]) < len(fakes[-1]):
        thres = len(real[0])

    else:
        thres = len(fakes[-1])

    for i in range(0, thres-1):
        f = torch.reshape(fakes[-1][i], (-1, 3, size, size))
        r = torch.reshape(real[0][i], (-1, 3, size, size))
        r_norm = (r - r.min()) / (r.max() - r.min())
        f_norm = (f - f.min()) / (f.max() - f.min())

        p.append(psnr(r_norm, f_norm))
        s.append(ssim(r_norm, f_norm))
        h.append(haar(r_norm, f_norm))
        ms.append(ms_gmsd(r_norm, f_norm))
        md.append(mdsi(r_norm, f_norm))

    with open(out + 'metrics_report.out', 'a') as mr_out:

        print('PSNR: {}, SSIM: {}, HAAR: {}, MSGMSD: {}, MDSI: {}'.format(
            sum(p)/(len(p)), sum(s)/(len(s)), sum(h)/(len(h)), sum(ms)/(len(ms)), sum(md)/(len(md)), file=mr_out))

    return 0
