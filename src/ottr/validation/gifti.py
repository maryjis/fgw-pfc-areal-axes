"""Minimal GIFTI reader (no nibabel in the analysis environment): data arrays (base64, optional gzip; float32 / int32 / uint8) and the label table."""
import base64, zlib
import xml.etree.ElementTree as ET
import numpy as np

DT = {'NIFTI_TYPE_FLOAT32': np.float32, 'NIFTI_TYPE_INT32': np.int32, 'NIFTI_TYPE_UINT8': np.uint8}


def read(path):
    root = ET.parse(path).getroot(); arrays = []
    for da in root.iter('DataArray'):
        dims = [int(da.get(f'Dim{i}')) for i in range(int(da.get('Dimensionality')))]; raw = base64.b64decode(da.find('Data').text.strip())
        if 'gzip' in da.get('Encoding').lower(): raw = zlib.decompress(raw)
        a = np.frombuffer(raw, dtype=DT[da.get('DataType')])
        if da.get('Endian') == 'BigEndian': a = a.byteswap()
        arrays.append((da.get('Intent'), a.reshape(dims, order='F' if da.get('ArrayIndexingOrder') == 'ColumnMajorOrder' else 'C')))
    labels = {int(l.get('Key')): (l.text or '').strip() for l in root.iter('Label')}
    return arrays, labels
