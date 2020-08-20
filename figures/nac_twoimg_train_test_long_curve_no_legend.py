import pickle
from pickle2excel.ysh_add import MyData

# file_dir = 'resnet_aug_l2_set12_g_train_test_curve2.pickle'
file_dir = 'long_train_curve/resnet_aug_l2_set12_g_twoimgs_curve2.pickle'
with open(file_dir, 'rb') as f:
    data_our = pickle.load(f)


psnr_list = data_our['current_psnr_record']
ssim_list = data_our['ssim_curve_max_record']
loss_list = data_our['training_loss_record']
psnr_1 = psnr_list[0:1001]
psnr_2 = psnr_list[10001:11002]

ssim_1 = ssim_list[0:1001]
ssim_2 = ssim_list[10001:11002]

loss_1 = loss_list[0:1001]
loss_2 = loss_list[10001:11002]

assert len(loss_1) == len(psnr_1)
assert len(loss_2) == len(psnr_2)

assert len(psnr_1) == len(ssim_1)
assert len(psnr_2) == len(ssim_2)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



#
# plt.figure(figsize=(6, 3))
fig, ax1 = plt.subplots(figsize=(6, 3.7))
plt.grid(axis="y")
plt.grid(ls='--')
# t = np.arange(0.01, 10.0, 0.01)
t = [i for i in range(len(psnr_1))]

s1 = loss_1
s3 = loss_2
psnr_color = '#0f0f0f'
# psnr_color2= '#e89119'
psnr_color2= '#f7b05d'
psnr_color22 = '#ea7f04'
h1,  = ax1.plot(t, s1, '-', color=psnr_color2, markersize=2, label='loss1')
h3,  = ax1.plot(t, s3, '-', color=psnr_color22, markersize=2, label='loss2')

# ax1.set_xlabel('Epochs ', fontsize=20)
# Make the y-axis label, ticks and tick labels match the line color.
# ax1.set_ylabel('Loss', color=psnr_color,fontsize=24)
ax1.tick_params('y', labelsize=18)
ax1.tick_params('x', labelsize=18)
ax1.set_ylim(0, 0.20)
# ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# ax1.legend('s1','s2')

# ssim_color = '#071199'
# ssim_color2 = '#0b6ce2'
ssim_color = '#071199'
ssim_color2 = '#68a4f2'
ssim_color22 = '#023be2'
ax2 = ax1.twinx()
s2 = psnr_1
s4 = psnr_2
# plt.grid(axis="y")
# plt.grid(ls='--')
# s2 = np.sin(2 * np.pi * t)
h2, =  ax2.plot(t, s2, f'-', color=ssim_color2, markersize=2, label='psnr1')
h4, =  ax2.plot(t, s4, f'-', color=ssim_color22, markersize=2, label='psnr2')
# # for i_x, i_y in zip(t, s2):
# #     plt.text(i_x+0.1, i_y, '{:.4f}'.format(i_y),ha='center',va='bottom', fontsize=12)
# ax2.set_ylabel('PSNR (dB)', fontsize=24)
ax2.tick_params('y', labelsize=18)
ax2.set_ylim(0, 40)

# plt.gca().xaxis.set_major_locator(ticker.FixedLocator([1,2,4,8]))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(8))
# ax1.legend([h1, h3],['Loss, Cameraman','Loss, House '], loc=(.06,.81))
# ax2.legend([h2, h4],['PSNR, Cameraman','PSNR, House'], loc =(.52,.81))

# plt.xticks(t,[1,2,4,8])
# ax1.set_xlim(0, 5)
# pylab.legend(loc='upper left')
# ax1.set_facecolor('#DAE8FC')
# fig.tight_layout()
plt.show()
# f = open('test.png')
# plt.rcParams['savefig.dpi'] = 100
# plt.rcParams['figure.dpi'] = 200

plt.rcParams['figure.figsize'] = (1.0, 0.61) # 设置figure_size尺寸
# plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
# plt.rcParams['image.cmap'] = 'gray'

fig.savefig('NAC_twoimgs_train_test_1w_curves_subimg_2000.png')
