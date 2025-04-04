import matplotlib.pyplot as plt
import numpy as np
import torch
from engine.loss import MSE_SAM_loss
#from datautils.mutihsi import MultisourceHSI, MultisourceHSIDataset
from engine.model import SpectralSharedEncoder
from downstream_classification import read_botswana, read_indian_pines
from collections import OrderedDict

# remove_bands = list(range(103,108))+list(range(149,163))+[119]
# hsi = loadmat('clfdata/IN/IndianPine.mat')['input']
# W,H,C = hsi.shape
# hsi = hsi.reshape(-1, C)
# hsi = minmax_scale(hsi)
# hsi = hsi.reshape(W,H,C)
# hsi = hsi.reshape(-1,C)
#
# waves = pd.read_csv('clfdata/IN/Calibration_Information_for_220_Channel_Data_Band_Set.csv')
# waves = np.array(waves.iloc[:,1])[:220]
# waves = np.delete(waves,remove_bands)
# waves_zeros = np.zeros_like(waves)
# data = torch.tensor(hsi[:1000]).unsqueeze(1).float()
# waves = torch.tensor(waves[np.newaxis,:].repeat(1000,axis=0)).float()
# waves_zeros = torch.tensor(waves_zeros[np.newaxis,:].repeat(1000,axis=0)).float()
#
ref_spec = []
ref_wave = []
with open('envi_plot.txt') as f:
    lines = f.readlines()
    lines = lines[3:]
    for line in lines:
        ref_spec.append(float(line.rstrip().split()[1]))
        ref_wave.append(float(line.rstrip().split()[0]) * 1000)

stride = 12

ref_spec = ref_spec[0:-1:stride]
ref_wave = np.asarray(ref_wave[0:-1:stride])

hsi, label, waves = read_botswana('clfdata/Botswana/Botswana.mat',
                                  'clfdata/Botswana/Botswana_gt.mat',
                                  'clfdata/Botswana/wave.csv')

#
# hsi, label, waves = read_indian_pines(
#     'clfdata/IN/IndianPine.mat', 'clfdata/IN/Indian_pines_gt.mat',
#
#     'clfdata/IN/Calibration_Information_for_220_Channel_Data_Band_Set.csv')
water = np.where(label == 1)
grass = np.where(label == 2)
water_specs = np.asarray(hsi[water])
grass_specs = np.asarray(hsi[grass])
# W, H, C = hsi.shape
# hsi = hsi.reshape(-1, C)
# np.random.shuffle(hsi)

# l = MSE_SAM_loss(grass_specs.mean(axis=0), ref_wave)

plt.plot(grass_specs.mean(axis=0), label='grass')
plt.plot(ref_spec, label='ref')
plt.legend()
plt.show()

ref_spec = torch.tensor(ref_spec).unsqueeze(0).unsqueeze(0).float()
ref_wave = torch.tensor(ref_wave).unsqueeze(0).float()

data = torch.tensor(grass_specs).unsqueeze(1).float()
data_water = torch.tensor(water_specs).unsqueeze(1).float()
waves_water = torch.tensor(waves[np.newaxis, :].repeat(data_water.shape[0], axis=0)).float()

waves = torch.tensor(waves[np.newaxis, :].repeat(data.shape[0], axis=0)).float()

model = SpectralSharedEncoder(embedding_dim=256,
                              num_heads=8,
                              decoder_depth=4,
                              encoder_depth=8,
                              )

# ckpt = torch.load('10_base_mask95_checkpoint.pt')
# weights = OrderedDict()
# for k, v in ckpt['model'].items():
#     name = k[7:]
#     weights[name] = v
# model.load_state_dict(weights)

if torch.cuda.is_available():
    model.cuda()
    data = data.cuda()
    data_water = data_water.cuda()
    waves = waves.cuda()
    waves_water = waves_water.cuda()
    ref_spec = ref_spec.cuda()
    ref_wave = ref_wave.cuda()

with torch.no_grad():
    h, x_hat = model(data, waves, 0.0)
    loss = MSE_SAM_loss(x_hat, data)
    print(loss.item())

    h_water, x_hat = model(data_water, waves_water, 0.0)
    loss = MSE_SAM_loss(x_hat, data_water)
    print(loss.item())

    h_ref, x_ref = model(ref_spec, ref_wave, 0.0)
    loss = MSE_SAM_loss(x_ref, ref_spec)
    print(loss.item())

h = h.detach().cpu().squeeze()
h_water = h_water.detach().cpu().squeeze()
h_ref = h_ref.detach().cpu().squeeze()

print(MSE_SAM_loss(h, h_ref))
print(MSE_SAM_loss(h_water, h_ref))



# plt.plot(torch.mean(h,dim=0), label='grass')
# plt.plot(torch.mean(h_water, dim=0), label='water')
plt.figure(figsize=(30, 8))
plt.plot(h_water.T, color='green')
plt.plot(h.T, color='blue')
plt.plot(h_ref, label='weed_ref', color='red')
plt.legend()
plt.ylim(-10,10)
# plt.ylim(0,12)
plt.show()
# plt.plot(data_water.detach().cpu().squeeze()[170], label='real')
# plt.plot(x_hat.detach().cpu().squeeze()[170], label='pred')
# plt.legend()
# plt.show()


# data_sam = []
#
# for i in h:
#     for j in h:
#         data_sam.append(MSE_SAM_loss(i, j))
# data_sam = list(np.random.rand(40000))
# plt.plot(h.detach().cpu().numpy().squeeze()[107, :], label='H1')
# plt.plot(h.detach().cpu().numpy().squeeze()[805, :], label='H2')
# plt.plot(data.detach().cpu().numpy().squeeze()[107, :], label='O1')
# plt.plot(data.detach().cpu().numpy().squeeze()[805, :], label='O2')


# plt.plot(h_sam)
# plt.scatter(h_sam, data_sam)
# plt.legend()
# plt.ylim((0, 1.0))
# plt.xlim((0, 1.0))
# plt.show()

a = 0

# N = 23
# waves = []
# bands = []
# with open('IN/Calibration_Information_for_220_Channel_Data_Band_Set.txt', 'r') as f:
#     for i in range(N - 1):
#         f.readline()
#     for line in f.readlines():
#         parts = line.split()
#         if len(parts) > 2:
#             try:
#                 waves.append(float(parts[2]))
#                 bands.append(int(parts[1]))
#             except ValueError:
#                 continue
# print(waves)
# clfdata = {
#     'bands':bands,
#     'waves':waves
# }
# df = pd.DataFrame(clfdata)
# df.to_csv('IN/Calibration_Information_for_220_Channel_Data_Band_Set.csv',index=False)
