import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
# from ysh_add import MyData
import pylab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#
# Set12，AWGN，\sigma=50，DnCNN：PSNR/SSIM = 27.18/0.7829
# Set12，Poisson，\lambda=50，ANSC:  PSNR/SSIM = 26.67/0.8434
# Set12，AWGN，\sigma=50，dip：PSNR/SSIM = 20.02/0.5769
# Set12，Poisson，\lambda=50，dip:  PSNR/SSIM = 19.16/0.594
# small_psnr_p = [32.52]
# small_psnr_g = [24.69]
# small_ssim_p = [0.9485]
# small_ssim_g = [0.8267]
small_psnr_p = [32.52]
small_psnr_g = [24.69]
small_ssim_p = [0.9485]
small_ssim_g = [0.8267]
# ours g, dncnn, dip g, ours p, nasc, dip
# small_psnr = [24.69, 27.18, 20.02, 32.52, 26.67, 19.16]
# small_ssim = [0.8267, 0.7829, 0.5769, 0.9485, 0.8434, 0.594]
# bm3d, dncnn, nac
# 26.72, 0.7676
small_psnr = [26.72, 27.18, 24.69]
small_ssim = [0.7676, 0.7829, 0.6680]
fig, ax1 = plt.subplots(figsize=(6, 3.7))
plt.grid(axis="y")
plt.grid(ls='--')
# t = np.arange(0.01, 10.0, 0.01)
t = [2,4,6]
s1 = small_psnr
# s11 = small_psnr_25
psnr_color = '#0f0f0f'
psnr_color2= '#f7b05d'
psnr_color22 = '#ea7f04'
# s1 = np.exp(t)
h1,  = ax1.plot(t, s1, 'o', color=psnr_color22, markersize=12, label='psnr_lambda15')
# h11,  = ax1.plot(t, s11, '-o', color=psnr_color22, markersize=12, label='psnr_lambda25')
# add coordinates
# for i_x, i_y in zip(t, s1):
#     plt.text(i_x, i_y-0.15, '{:.1f}'.format(i_y),ha='center',va='top', fontsize=12)

ax1.set_xlabel('Method', fontsize=18)
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('PSNR (dB)', color=psnr_color,fontsize=24)
ax1.tick_params('y', labelsize=18)
ax1.tick_params('x', labelsize=18)
ax1.set_ylim(14, 28)
# ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

ax1.legend('s1','s2')

ssim_color = '#071199'
ssim_color2 = '#68a4f2'
ssim_color22 = '#023be2'
ax2 = ax1.twinx()
s2 = small_ssim
# s22 = small_ssim_25
# plt.grid(axis="y")
# plt.grid(ls='--')
# s2 = np.sin(2 * np.pi * t)
h2, =  ax2.plot(t, s2, f'^', color=ssim_color22, markersize=12, label='ssim_lambda15')
# h22, =  ax2.plot(t, s22, f'-^', color=ssim_color22, markersize=12, label='ssim_lambda25')

# for i_x, i_y in zip(t, s2):
#     plt.text(i_x+0.1, i_y, '{:.4f}'.format(i_y),ha='center',va='bottom', fontsize=12)
ax2.set_ylabel('SSIM', fontsize=24)
ax2.tick_params('y', labelsize=18)
ax2.set_ylim(0.66, 0.80)

# plt.gca().xaxis.set_major_locator(ticker.FixedLocator([1,2,4,8]))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

ax1.legend([h1],['PSNR (dB), $\sigma$ = 50'], loc=(.01,.12))
# ax1.legend([h11],['PSNR (dB), sigma = 25'], loc=(.01,.3))

ax2.legend([h2],['SSIM, $\sigma$ = 50'], loc =(.01,.01))
# ax2.legend([h22],['SSIM, sigma =25'], loc =(.01,.1))
plt.xticks(t,['BM3D','DnCNN','NAC'])

# plt.xticks(t,['NAC-AWGN','DnCNN','DIP','NAC-POISSON','NASC','DIP'])
# ax1.set_xlim(0, 5)
# pylab.legend(loc='upper left')
# ax1.set_facecolor('#DAE8FC')
fig.tight_layout()
plt.show()

# plt.rcParams['figure.figsize'] = (1.0, 0.61) # 设置figure_size尺寸
fig.savefig('Large_sigma_final_22.pdf')
