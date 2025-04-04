import h5py
aa = ['DESIS-HSI-L2A-DT0491642772_013-20200827T073441-V0220',
 'DESIS-HSI-L2A-DT0778482596_002-20221014T050309-V0220',
 'DESIS-HSI-L2A-DT0842938904_035-20230406T181704-V0220',
 'DESIS-HSI-L2A-DT0884261736_007-20230728T123218-V0220',
 'DESIS-HSI-L2A-DT0886101060_002-20230802T005625-V0220']
bb = ['ENMap-HSI-L2A-DT0000001293_20220627T025923Z_006_V010402_20240722T030828Z',
 'ENMap-HSI-L2A-DT0000001614_20220708T184006Z_004_V010402_20240704T191633Z',
 'ENMap-HSI-L2A-DT0000001649_20220714T175819Z_001_V010402_20240704T181107Z',
 'ENMap-HSI-L2A-DT0000001868_20220724T025922Z_007_V010402_20240722T025712Z',
 'ENMap-HSI-L2A-DT0000002057_20220729T192659Z_033_V010402_20240703T084031Z']
with h5py.File('MultiSourceHSI.hdf5', "a") as f:
    for i in bb:
        del f['ENMap'][i]

# datasets = [g[key] for key in g.keys()]
# length = sum([d.attrs['length'] for d in datasets])
# start = 0
# end = 0
#
# length_intervals = []
# lengths = [0] + [g[key].attrs['length'] for key in g.keys()]
# for i in range(len(lengths) - 1):
#     start = start + lengths[i]
#     end = end + lengths[i + 1]
#     length_intervals.append((start, end - 1))
#
# a = 0