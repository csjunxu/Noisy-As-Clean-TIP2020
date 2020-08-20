import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from ysh_add import MyData
import pylab


small_psnr = [39.99, 36.55, 34.24, 32.46, 31.08]
small_ssim = [0.9820, 0.9569, 0.9277, 0.8961,0.8654]
small_psnr2 = [40.83,37.63,35.28,32.96,32.08]
small_ssim2 = [0.9857,0.9642,0.9386,0.9101,0.8879]


# small_psnr2 = []
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#
# plt.figure(figsize=(6, 3))
fig, ax1 = plt.subplots(figsize=(6, 3.7))
plt.grid(axis="y")
plt.grid(ls='--')
# t = np.arange(0.01, 10.0, 0.01)
t = [1, 2,3,4,5]
s1 = small_psnr2
s11 = small_psnr
psnr_color = '#0f0f0f'
psnr_color2= '#f7b05d'
psnr_color22 = '#ea7f04'
# s1 = np.exp(t)
h1,  = ax1.plot(t, s1, '-o', color=psnr_color2, markersize=12, label='psnr_oracle')
h11,  = ax1.plot(t, s11, '-o', color=psnr_color22, markersize=12, label='psnr_ours')

# for i_x, i_y in zip(t, s1):
#     plt.text(i_x, i_y-0.15, '{:.1f}'.format(i_y),ha='center',va='top', fontsize=12)
ax1.set_xlabel(' AWGN Noise Level', fontsize=17)
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('PSNR (dB)', color=psnr_color,fontsize=24)
ax1.tick_params('y', labelsize=18)
ax1.tick_params('x', labelsize=18)
ax1.set_ylim(28, 48)
# ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

ax1.legend('s1','s2')

ssim_color = '#071199'
ssim_color2 = '#68a4f2'
ssim_color22 = '#023be2'
ax2 = ax1.twinx()
s2 = small_ssim2
s22 = small_ssim
# plt.grid(axis="y")
# plt.grid(ls='--')
# s2 = np.sin(2 * np.pi * t)
h2, =  ax2.plot(t, s2, f'-^', color=ssim_color2, markersize=12, label='ssim_oracle')
h22, =  ax2.plot(t, s22, f'-^', color=ssim_color22, markersize=12, label='ssim_ours')
# for i_x, i_y in zip(t, s2):
#     plt.text(i_x+0.1, i_y, '{:.4f}'.format(i_y),ha='center',va='bottom', fontsize=12)
ax2.set_ylabel('SSIM', fontsize=24)
ax2.tick_params('y', labelsize=18)
ax2.set_ylim(0.80, 1)

# plt.gca().xaxis.set_major_locator(ticker.FixedLocator([1,2,4,8]))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
ax1.legend([h1, h11],['PSNR (dB), Oracle','PSNR (dB), Ours'], loc=(.01,.19))
ax2.legend([h2, h22],['SSIM, Oracle', 'SSIM, Ours'], loc =(.01,.01))
plt.xticks(t,[5,10,15,20,25])
# ax1.set_xlim(0, 5)
# pylab.legend(loc='upper left')
# ax1.set_facecolor('#DAE8FC')
fig.tight_layout()
plt.show()
# f = open('test.png')
# plt.rcParams['savefig.dpi'] = 100
# plt.rcParams['figure.dpi'] = 200

plt.rcParams['figure.figsize'] = (1.0, 0.61) # 设置figure_size尺寸
# plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
# plt.rcParams['image.cmap'] = 'gray'

fig.savefig('set12_g_oracle_final3.pdf')
