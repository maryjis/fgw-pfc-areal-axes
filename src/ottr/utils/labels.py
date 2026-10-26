"""Label tables shared by all modules: fine region -> coarse area of the adult PFC, layers, labels to ignore, marker-gene
programmes used to name an axis by the genes at its two ends."""
import re

LAYERS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
COL_LAYERS = ['L2', 'L3', 'L4', 'L5', 'L6']        # layers that make up a column profile
BAD_REGION = {'nan', 'None', 'remove', 'unassigned', 'WM', 'white_matter'}

# fine region (atlas area) -> coarse area. Seen in T419/T420 (majority column area there): A11 A13L A13M A13b -> OPFC,
# A14R A32 A32V -> MPFC, A45 A47L A47M -> VLPFC, A8aD A8b -> DLPFC, A8Av -> Premotor; the rest by atlas position.
AREA = {'A11': 'OPFC', 'A13L': 'OPFC', 'A13M': 'OPFC', 'A13a': 'OPFC', 'A13b': 'OPFC', 'A47O': 'OPFC', 'OPAl': 'OPFC', 'OPro': 'OPFC',
        'A14R': 'MPFC', 'A14C': 'MPFC', 'A32': 'MPFC', 'A32V': 'MPFC', 'A25': 'MPFC',
        'A45': 'VLPFC', 'A47L': 'VLPFC', 'A47M': 'VLPFC', 'A46V': 'VLPFC',
        'A8aD': 'DLPFC', 'A8b': 'DLPFC', 'A8C': 'DLPFC', 'A9': 'DLPFC', 'A46D': 'DLPFC',
        'A8Av': 'Premotor', 'A6DR': 'Premotor', 'A6Va': 'Premotor', 'A6Vb': 'Premotor', 'ProM': 'Premotor',
        'A24a': 'ACC', 'A24b': 'ACC', 'A24c': 'ACC', 'A10': 'FPC'}
SA_ANNO = {'DLPFC': 'A', 'MPFC': 'unsure', 'OPFC': 'unsure', 'VLPFC': 'S_like', 'Premotor': 'S_like', 'ACC': 'remove', 'FPC': 'unsure'}
# adult coarse area -> the four areas annotated in the young brains (premotor is drawn with dlPFC but is not in the young data)
COARSE_DEV = {'MPFC': 'mPFC', 'ACC': 'mPFC', 'OPFC': 'oPFC', 'VLPFC': 'vlPFC', 'DLPFC': 'dlPFC', 'Premotor': 'dlPFC'}
DEV_AREAS = ['mPFC', 'oPFC', 'vlPFC', 'dlPFC']

# marker programmes for naming gene ends of an axis (first match with >= 2 genes wins, in this order)
MARK = [('interneurons', {'GAD1', 'GAD2', 'SLC32A1', 'SST', 'VIP', 'CRH', 'CALB2', 'SLC6A1', 'PVALB', 'LAMP5'}),
        ('meninges/vascular', {'DCN', 'PRG4', 'IGF2', 'IGFBP7', 'APOD', 'LGALS1', 'MFAP5', 'SLC22A2', 'COL1A2', 'PTGDS', 'COL1A1', 'LUM'}),
        ('myelin', {'MBP', 'PLP1', 'MOBP', 'MOG', 'TF', 'CNP', 'MAG', 'BCAS1'}),
        ('glia', {'SLC1A2', 'DOCK3', 'MT3', 'NDRG2', 'GLUL', 'CCDC3', 'SLC1A3', 'GFAP', 'AQP4', 'GJA1', 'CLU'}),
        ('deep L3 / L4 (RORB)', {'RORB', 'CNTN5', 'DCC', 'PLCH1', 'PTPRT'}),
        ('upper L2/3 (CAMK2N1)', {'CAMK2N1', 'RGS4', 'WFS1', 'OCM2', 'LRRTM4', 'CUX2', 'CXCL14', 'RASGRF2'}),
        ('deep L5-6', {'FEZF2', 'TLE4', 'HS3ST4', 'NXPH3', 'TBR1', 'FOXP2', 'KRT17', 'SEMA3E', 'CPLX3'}),
        ('activity genes', {'FOS', 'EGR1', 'NPAS4', 'NR4A1', 'ARC', 'JUNB', 'IER2'}),
        ('neuronal', {'NRGN', 'SNAP25', 'CHN1', 'VSNL1', 'ENC1', 'NEFM', 'NEFL', 'STMN2', 'LMO4', 'SLC17A7', 'SYT1'}),
        ('ribosomal', None)]
KNOWN = {k for k, _ in MARK}


def genes_tag(gs):
    """Name of a gene list ('a, b, c'): 'ribosomal' with >= 3 RPL/RPS genes, else the marker programme sharing >= 2 genes,
    else the first two genes."""
    gs = [g.strip() for g in str(gs).split(',') if g.strip()]
    if sum(bool(re.match(r'^RP[LS]\d', g)) for g in gs) >= 3: return 'ribosomal'
    best = max(MARK, key=lambda kv: 0 if kv[1] is None else len(kv[1] & set(gs)))
    return best[0] if best[1] and len(best[1] & set(gs)) >= 2 else ', '.join(gs[:2])


def section_number(s):
    """'T417' -> 417 (section order of the serial series)."""
    return int(s[1:])
