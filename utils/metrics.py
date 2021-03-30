import numpy as np
import torch

from piqa import SSIM, HaarPSI, PSNR, MS_SSIM, MS_GMSD, MDSI


def compute_metrics(real, fakes, out):

    p, s, h, ms, md = 0, 0, 0, 0, 0

    ssim = SSIM().cpu()
    psnr = PSNR()
    haar = HaarPSI()
    msssim = MS_SSIM()
    ms_gmsd = MS_GMSD()
    mdsi = MDSI()

    if len(real[0]) > len((fakes[-1]):
        thres=len(real[0])

    elif len(real[0]) < len((fakes[-1]):
        thres=len(fakes[-1])

    else:
        thres=len(fakes[-1])

    for i in range(0, thres-1)):
        f=torch.reshape(fakes[-1][i], (-1, 3, 64, 64))
        r=torch.reshape(real[0][i], (-1, 3, 64, 64))
        r_norm=(r - r.min()) / (r.max() - r.min())
        f_norm=(f - f.min()) / (f.max() - f.min())

        p += psnr(r_norm, f_norm)
        s += ssim(r_norm, f_norm)
        h += haar(r_norm, f_norm)
        ms += ms_gmsd(r_norm, f_norm)
        md += mdsi(r_norm, f_norm)

    with open(out + 'metrics_report.out') as mr_out:

        print('PSNR: {}, SSIM: {}, HAAR: {}, MSGMSD: {}, MDSI: {}'.format(
                p/(len(p)), s/(len(s)), h/(len(h)), ms/(len(ms)), md/(len(md)), file=mr_out))

    return 0
