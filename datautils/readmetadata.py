import xml.etree.ElementTree as ET
import pandas as pd
import os

def readcenterwavelength(path):
    tree = ET.parse(path)
    root = tree.getroot()
    bandCharacterisation = root.find('specific').find('bandCharacterisation')
    return [child.find('wavelengthCenterOfBand').text for child in bandCharacterisation]

def readgf5centerwavelength(path):
    with open(os.path.join(path,'GF5_AHSI_SWIR_Spectralresponse.raw'),'r') as f:
        SWIR = [float(line.split('\t')[0]) for line in f.readlines()]
    with open(os.path.join(path,'GF5_AHSI_VNIR_Spectralresponse.raw'),'r') as f:
        VNIR = [float(line.split('\t')[0]) for line in f.readlines()]
    VNSW = VNIR+SWIR
    VNSW.sort()
    print(VNSW)

if __name__ == '__main__':
    # df = pd.read_csv(r"E:\Dataset\GF5\GF5_AHSI_E110.53_N32.30_20191229_008729_L10000069228\GF5_AHSI_SWIR_Spectralresponse.raw",header=None)
    # a = 0
    # with open(r'E:\Dataset\GF5\GF5_AHSI_E110.53_N32.30_20191229_008729_L10000069228\GF5_AHSI_SWIR_Spectralresponse.raw','r') as f:
    #     # d = f.readlines()
    #     d = [float(line.split('\t')[0]) for line in f.readlines()]
    #
    # print(d)
    readgf5centerwavelength(r'E:\Dataset\GF5\GF5_AHSI_E110.53_N32.30_20191229_008729_L10000069228')