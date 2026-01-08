"""
Slater-Koster Hopping Integrals for s, p, d, and f orbitals.

This module implements the two-center Slater-Koster integrals for tight-binding
Hamiltonians. The formulas are based on:
  - Slater & Koster, Phys. Rev. 94, 1498 (1954) - original s, p, d formulas
  - Takegahara et al., J. Phys. C 13, 583 (1980) - f orbital extensions
  - Sharma, Phys. Rev. B 19, 2813 (1979) - complete f orbital tables

Orbital ordering (17 orbitals total):
  Index 0: s
  Index 1-3: px, py, pz
  Index 4-8: dxy, dyz, dxz, dx2-y2, dz2
  Index 9-15: fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)
  Index 16: S (excited s orbital)

Direction cosines: l, m, n are cosines of bond direction with x, y, z axes.
  l = cos(angle with x-axis)
  m = cos(angle with y-axis)
  n = cos(angle with z-axis)
  where l^2 + m^2 + n^2 = 1

Hopping parameter naming convention:
  V_αβγ where α, β are orbital types and γ is the bond symmetry (σ, π, δ, φ)
  σ (sigma): m_l = 0 along bond axis
  π (pi): |m_l| = 1 along bond axis
  δ (delta): |m_l| = 2 along bond axis
  φ (phi): |m_l| = 3 along bond axis

Author: Santosh Kumar Radha
"""

from numpy import sqrt


def get_hop_int(
    # s-s interaction
    V_sss=0,
    # s-p interaction
    V_sps=0,
    # p-p interactions
    V_pps=0,
    V_ppp=0,
    # s-d interaction
    V_sds=0,
    # p-d interactions
    V_pds=0,
    V_pdp=0,
    # d-d interactions
    V_dds=0,
    V_ddp=0,
    V_ddd=0,
    # s-f interaction
    V_sfs=0,
    # p-f interactions
    V_pfs=0,
    V_pfp=0,
    # d-f interactions
    V_dfs=0,
    V_dfp=0,
    V_dfd=0,
    # f-f interactions
    V_ffs=0,
    V_ffp=0,
    V_ffd=0,
    V_fff=0,
    # S (excited s) interactions
    V_SSs=0,
    V_sSs=0,
    V_Sps=0,
    V_Sds=0,
    V_Sfs=0,
    # Direction cosines
    l=0,
    m=0,
    n=0
):
    """
    Calculate Slater-Koster hopping integrals for all orbital pairs.

    Parameters
    ----------
    V_sss, V_sps, ... : float
        Slater-Koster hopping parameters for each orbital pair and bond type.
    l, m, n : float
        Direction cosines of the bond vector (must satisfy l² + m² + n² = 1).

    Returns
    -------
    hop_int : list of lists
        17×17 matrix of hopping integrals between all orbital pairs.
    """
    # Initialize 17×17 hopping matrix
    N_ORBITALS = 17
    hop_int = [[0.0 for _ in range(N_ORBITALS)] for __ in range(N_ORBITALS)]

    # Precompute common terms for efficiency
    l2, m2, n2 = l**2, m**2, n**2
    lm, mn, nl = l * m, m * n, n * l
    lmn = l * m * n

    # Useful combinations
    l2_m2 = l2 - m2  # (l² - m²)
    l2_p_m2 = l2 + m2  # (l² + m²)

    # Square roots
    sqrt3 = sqrt(3)
    sqrt5 = sqrt(5)
    sqrt15 = sqrt(15)

    # =========================================================================
    # s-s interaction (index 0-0)
    # =========================================================================
    hop_int[0][0] = V_sss

    # =========================================================================
    # s-p interactions (index 0 with 1,2,3)
    # H(s, p_i) = l_i * V_spσ where l_i is direction cosine along i
    # =========================================================================
    hop_int[0][1] = l * V_sps  # s-px
    hop_int[0][2] = m * V_sps  # s-py
    hop_int[0][3] = n * V_sps  # s-pz
    # Transpose with sign (p orbital is odd under inversion)
    hop_int[1][0] = -hop_int[0][1]
    hop_int[2][0] = -hop_int[0][2]
    hop_int[3][0] = -hop_int[0][3]

    # =========================================================================
    # s-d interactions (index 0 with 4,5,6,7,8)
    # =========================================================================
    hop_int[0][4] = sqrt3 * lm * V_sds  # s-dxy
    hop_int[0][5] = sqrt3 * mn * V_sds  # s-dyz
    hop_int[0][6] = sqrt3 * nl * V_sds  # s-dxz
    hop_int[0][7] = sqrt3 / 2.0 * l2_m2 * V_sds  # s-dx2-y2
    hop_int[0][8] = (n2 - 0.5 * l2_p_m2) * V_sds  # s-dz2
    # Transpose (d orbitals are even, same sign)
    for i in range(4, 9):
        hop_int[i][0] = hop_int[0][i]

    # =========================================================================
    # s-f interactions (index 0 with 9,10,11,12,13,14,15)
    # Formulas from Takegahara et al. and Sharma
    # =========================================================================
    # fz3: proportional to n(5n² - 3)
    hop_int[0][9] = n * (5 * n2 - 3) / 2.0 * V_sfs

    # fxz2: proportional to l(5n² - 1)
    hop_int[0][10] = sqrt(3.0 / 8.0) * l * (5 * n2 - 1) * V_sfs

    # fyz2: proportional to m(5n² - 1)
    hop_int[0][11] = sqrt(3.0 / 8.0) * m * (5 * n2 - 1) * V_sfs

    # fz(x2-y2): proportional to n(l² - m²)
    hop_int[0][12] = sqrt15 / 2.0 * n * l2_m2 * V_sfs

    # fxyz: proportional to lmn
    hop_int[0][13] = sqrt15 * lmn * V_sfs

    # fx(x2-3y2): proportional to l(l² - 3m²)
    hop_int[0][14] = sqrt(5.0 / 8.0) * l * (l2 - 3 * m2) * V_sfs

    # fy(3x2-y2): proportional to m(3l² - m²)
    hop_int[0][15] = sqrt(5.0 / 8.0) * m * (3 * l2 - m2) * V_sfs

    # Transpose (f orbitals are odd, opposite sign)
    for i in range(9, 16):
        hop_int[i][0] = -hop_int[0][i]

    # =========================================================================
    # p-p interactions (indices 1,2,3 with 1,2,3)
    # =========================================================================
    hop_int[1][1] = l2 * V_pps + (1.0 - l2) * V_ppp  # px-px
    hop_int[2][2] = m2 * V_pps + (1.0 - m2) * V_ppp  # py-py
    hop_int[3][3] = n2 * V_pps + (1.0 - n2) * V_ppp  # pz-pz

    hop_int[1][2] = lm * (V_pps - V_ppp)  # px-py
    hop_int[2][1] = hop_int[1][2]

    hop_int[1][3] = nl * (V_pps - V_ppp)  # px-pz
    hop_int[3][1] = hop_int[1][3]

    hop_int[2][3] = mn * (V_pps - V_ppp)  # py-pz
    hop_int[3][2] = hop_int[2][3]

    # =========================================================================
    # p-d interactions (indices 1,2,3 with 4,5,6,7,8)
    # =========================================================================
    # px-d
    hop_int[1][4] = sqrt3 * l2 * m * V_pds + m * (1.0 - 2 * l2) * V_pdp
    hop_int[1][5] = lmn * (sqrt3 * V_pds - 2 * V_pdp)
    hop_int[1][6] = sqrt3 * l2 * n * V_pds + n * (1.0 - 2 * l2) * V_pdp
    hop_int[1][7] = 0.5 * sqrt3 * l * l2_m2 * V_pds + l * (1.0 - l2 + m2) * V_pdp
    hop_int[1][8] = l * (n2 - 0.5 * l2_p_m2) * V_pds - sqrt3 * l * n2 * V_pdp

    # py-d
    hop_int[2][4] = sqrt3 * m2 * l * V_pds + l * (1.0 - 2 * m2) * V_pdp
    hop_int[2][5] = sqrt3 * m2 * n * V_pds + n * (1.0 - 2 * m2) * V_pdp
    hop_int[2][6] = lmn * (sqrt3 * V_pds - 2 * V_pdp)
    hop_int[2][7] = 0.5 * sqrt3 * m * l2_m2 * V_pds - m * (1.0 + l2 - m2) * V_pdp
    hop_int[2][8] = m * (n2 - 0.5 * l2_p_m2) * V_pds - sqrt3 * m * n2 * V_pdp

    # pz-d
    hop_int[3][4] = lmn * (sqrt3 * V_pds - 2 * V_pdp)
    hop_int[3][5] = sqrt3 * n2 * m * V_pds + m * (1.0 - 2 * n2) * V_pdp
    hop_int[3][6] = sqrt3 * n2 * l * V_pds + l * (1.0 - 2 * n2) * V_pdp
    hop_int[3][7] = 0.5 * sqrt3 * n * l2_m2 * V_pds - n * l2_m2 * V_pdp
    hop_int[3][8] = n * (n2 - 0.5 * l2_p_m2) * V_pds + sqrt3 * n * l2_p_m2 * V_pdp

    # Transpose (opposite sign for p-d)
    for i in range(1, 4):
        for j in range(4, 9):
            hop_int[j][i] = -hop_int[i][j]

    # =========================================================================
    # p-f interactions (indices 1,2,3 with 9,10,11,12,13,14,15)
    # Formulas from Takegahara et al., J. Phys. C 13, 583 (1980)
    # =========================================================================
    # Precompute more useful terms
    five_n2_1 = 5 * n2 - 1
    five_n2_3 = 5 * n2 - 3

    # px-f interactions
    hop_int[1][9] = (sqrt(3.0 / 8.0) * l * five_n2_3 * V_pfs +
                    sqrt(3.0 / 2.0) * l * (1 - n2) * V_pfp)
    hop_int[1][10] = (0.25 * (5 * n2 - 1 - 4 * l2) * V_pfs +
                     0.5 * (l2 - n2 + (5 * n2 - 1) * l2) * V_pfp)
    hop_int[1][11] = (lm * (5 * n2 - 1) / 4.0 * V_pfs +
                     lm * (1 + 0.5 * (5 * n2 - 1)) * V_pfp)
    hop_int[1][12] = (sqrt(5.0 / 8.0) * l * n * l2_m2 * V_pfs +
                     sqrt(5.0 / 2.0) * l * n * (1 - 0.5 * l2_m2) * V_pfp)
    hop_int[1][13] = (sqrt(5.0 / 2.0) * mn * l2 * V_pfs +
                     sqrt(5.0 / 2.0) * mn * (l2 - 1) * V_pfp)
    hop_int[1][14] = (sqrt(15.0) / 8.0 * (l2 - 3 * m2 + 4 * l2 * (l2 - m2) / l2_m2 if abs(l2_m2) > 1e-10 else l2 - 3 * m2) * V_pfs +
                     sqrt(15.0) / 4.0 * (m2 - l2 * (l2 - 3 * m2) / (l2_m2 if abs(l2_m2) > 1e-10 else 1)) * V_pfp) if abs(l2_m2) > 1e-10 else 0
    hop_int[1][15] = (sqrt(5.0) / 4.0 * lm * (3 - 4 * m2 / (l2_p_m2 if l2_p_m2 > 1e-10 else 1)) * V_pfs +
                     sqrt(5.0) / 2.0 * lm * (2 * m2 / (l2_p_m2 if l2_p_m2 > 1e-10 else 1) - 1) * V_pfp) if l2_p_m2 > 1e-10 else 0

    # Simplified p-f formulas (using standard Sharma conventions)
    # px-f
    hop_int[1][9] = sqrt(3.0/8.0) * l * (5*n2 - 3) * V_pfs + sqrt(6.0)/2.0 * l * (1 - n2) * V_pfp
    hop_int[1][10] = 0.25 * (l2*(5*n2-1) - 2*(5*n2-1) + 4*l2) * V_pfs + 0.5 * (5*n2*l2 - l2 - n2 + l2) * V_pfp
    hop_int[1][11] = 0.25 * sqrt(3.0) * lm * (5*n2 - 1) * V_pfs + sqrt(3.0)/2.0 * lm * n2 * V_pfp
    hop_int[1][12] = sqrt(5.0/8.0) * nl * l2_m2 * V_pfs + sqrt(5.0/2.0) * nl * (n2 - 0.5*l2_m2) * V_pfp
    hop_int[1][13] = sqrt(5.0/2.0) * mn * l2 * V_pfs + sqrt(5.0/2.0) * mn * (l2 - 1) * V_pfp
    hop_int[1][14] = sqrt(15.0)/8.0 * (l2*(l2 - 3*m2)/l2_p_m2 if l2_p_m2 > 1e-10 else 0) * V_pfs + sqrt(15.0)/4.0 * (m2*(l2 - 3*m2)/l2_p_m2 if l2_p_m2 > 1e-10 else 0) * V_pfp if l2_p_m2 > 1e-10 else 0
    hop_int[1][15] = sqrt(5.0)/8.0 * lm * (3*l2 - m2)/l2_p_m2 * V_pfs - sqrt(5.0)/4.0 * lm * (l2 + 3*m2)/l2_p_m2 * V_pfp if l2_p_m2 > 1e-10 else 0

    # Corrected px-f using Sharma's Table I
    hop_int[1][9] = sqrt(6.0)/4.0 * l * (5*n2 - 1) * V_pfs - sqrt(6.0)/2.0 * l * n2 * V_pfp
    hop_int[1][10] = (3.0/4.0 * l2 * (5*n2 - 1) - (5*n2 - 1)/2.0) * V_pfs + (l2 * (1 - 5*n2/2.0) + n2) * V_pfp
    hop_int[1][11] = 3.0/4.0 * lm * (5*n2 - 1) * V_pfs - 5.0/2.0 * lm * n2 * V_pfp
    hop_int[1][12] = sqrt(10.0)/4.0 * nl * l2_m2 * V_pfs - sqrt(10.0)/2.0 * nl * (1 + l2_m2/2.0) * V_pfp
    hop_int[1][13] = sqrt(10.0)/2.0 * mn * l2 * V_pfs - sqrt(10.0)/2.0 * mn * (1 - l2) * V_pfp
    hop_int[1][14] = (sqrt(30.0)/8.0 * l * (l2 - 3*m2) * V_pfs -
                     sqrt(30.0)/4.0 * l * (l2 - 3*m2) * (1 - l2 + m2)/(l2_p_m2 if l2_p_m2 > 1e-10 else 1) * V_pfp) if l2_p_m2 > 1e-10 else 0
    hop_int[1][15] = (sqrt(10.0)/8.0 * m * (3*l2 - m2) * V_pfs +
                     sqrt(10.0)/4.0 * m * (1 - (3*l2 - m2)*(1 + l2 - m2)/(l2_p_m2 if l2_p_m2 > 1e-10 else 1)) * V_pfp) if l2_p_m2 > 1e-10 else 0

    # py-f (by symmetry l <-> m in many terms)
    hop_int[2][9] = sqrt(6.0)/4.0 * m * (5*n2 - 1) * V_pfs - sqrt(6.0)/2.0 * m * n2 * V_pfp
    hop_int[2][10] = 3.0/4.0 * lm * (5*n2 - 1) * V_pfs - 5.0/2.0 * lm * n2 * V_pfp
    hop_int[2][11] = (3.0/4.0 * m2 * (5*n2 - 1) - (5*n2 - 1)/2.0) * V_pfs + (m2 * (1 - 5*n2/2.0) + n2) * V_pfp
    hop_int[2][12] = sqrt(10.0)/4.0 * mn * l2_m2 * V_pfs + sqrt(10.0)/2.0 * mn * (1 - l2_m2/2.0) * V_pfp
    hop_int[2][13] = sqrt(10.0)/2.0 * nl * m2 * V_pfs - sqrt(10.0)/2.0 * nl * (1 - m2) * V_pfp
    hop_int[2][14] = (sqrt(30.0)/8.0 * m * (l2 - 3*m2) * V_pfs +
                     sqrt(30.0)/4.0 * m * (l2 - 3*m2) * (1 + l2 - m2)/(l2_p_m2 if l2_p_m2 > 1e-10 else 1) * V_pfp) if l2_p_m2 > 1e-10 else 0
    hop_int[2][15] = (sqrt(10.0)/8.0 * l * (3*l2 - m2) * V_pfs -
                     sqrt(10.0)/4.0 * l * (1 + (3*l2 - m2)*(1 - l2 + m2)/(l2_p_m2 if l2_p_m2 > 1e-10 else 1)) * V_pfp) if l2_p_m2 > 1e-10 else 0

    # pz-f
    hop_int[3][9] = (n2 * (5*n2 - 3) + 3.0/2.0 * (1 - n2)) * V_pfs + sqrt(6.0) * n2 * (1 - n2) * V_pfp
    hop_int[3][10] = sqrt(6.0)/4.0 * nl * (5*n2 - 3) * V_pfs + sqrt(6.0)/2.0 * nl * (1 - 5*n2 + 4*n2) * V_pfp
    hop_int[3][11] = sqrt(6.0)/4.0 * mn * (5*n2 - 3) * V_pfs + sqrt(6.0)/2.0 * mn * (1 - 5*n2 + 4*n2) * V_pfp
    hop_int[3][12] = sqrt(10.0)/4.0 * n2 * l2_m2 * V_pfs + sqrt(10.0)/4.0 * l2_m2 * (1 - 2*n2) * V_pfp
    hop_int[3][13] = sqrt(10.0)/2.0 * lm * n2 * V_pfs - sqrt(10.0)/2.0 * lm * (1 - n2) * V_pfp
    hop_int[3][14] = sqrt(30.0)/8.0 * nl * (l2 - 3*m2) * V_pfs + sqrt(30.0)/4.0 * nl * (1 - (l2 - 3*m2)/2.0) * V_pfp
    hop_int[3][15] = sqrt(10.0)/8.0 * mn * (3*l2 - m2) * V_pfs + sqrt(10.0)/4.0 * mn * (1 - (3*l2 - m2)/2.0) * V_pfp

    # Transpose (p and f both odd -> same sign)
    for i in range(1, 4):
        for j in range(9, 16):
            hop_int[j][i] = hop_int[i][j]

    # =========================================================================
    # d-d interactions (indices 4,5,6,7,8 with 4,5,6,7,8)
    # =========================================================================
    # dxy-dxy, dyz-dyz, dxz-dxz
    hop_int[4][4] = (l2 * m2 * (3 * V_dds - 4 * V_ddp + V_ddd) +
                    (l2 + m2) * V_ddp + n2 * V_ddd)
    hop_int[5][5] = (m2 * n2 * (3 * V_dds - 4 * V_ddp + V_ddd) +
                    (m2 + n2) * V_ddp + l2 * V_ddd)
    hop_int[6][6] = (n2 * l2 * (3 * V_dds - 4 * V_ddp + V_ddd) +
                    (n2 + l2) * V_ddp + m2 * V_ddd)

    # Cross terms between t2g orbitals
    hop_int[4][5] = l * m2 * n * (3 * V_dds - 4 * V_ddp + V_ddd) + nl * (V_ddp - V_ddd)
    hop_int[4][6] = n * l2 * m * (3 * V_dds - 4 * V_ddp + V_ddd) + mn * (V_ddp - V_ddd)
    hop_int[5][6] = m * n2 * l * (3 * V_dds - 4 * V_ddp + V_ddd) + lm * (V_ddp - V_ddd)
    hop_int[5][4] = hop_int[4][5]
    hop_int[6][4] = hop_int[4][6]
    hop_int[6][5] = hop_int[5][6]

    # t2g - eg cross terms
    hop_int[4][7] = 0.5 * lm * l2_m2 * (3 * V_dds - 4 * V_ddp + V_ddd)
    hop_int[5][7] = 0.5 * mn * (l2_m2 * (3 * V_dds - 4 * V_ddp + V_ddd) - 2 * (V_ddp - V_ddd))
    hop_int[6][7] = 0.5 * nl * (l2_m2 * (3 * V_dds - 4 * V_ddp + V_ddd) + 2 * (V_ddp - V_ddd))

    hop_int[7][4] = hop_int[4][7]
    hop_int[7][5] = hop_int[5][7]
    hop_int[7][6] = hop_int[6][7]

    hop_int[4][8] = sqrt3 * (lm * (n2 - 0.5 * l2_p_m2) * V_dds -
                            2 * lm * n2 * V_ddp + 0.5 * lm * (1.0 + n2) * V_ddd)
    hop_int[5][8] = sqrt3 * (mn * (n2 - 0.5 * l2_p_m2) * V_dds +
                            mn * (l2_p_m2 - n2) * V_ddp - 0.5 * mn * l2_p_m2 * V_ddd)
    hop_int[6][8] = sqrt3 * (nl * (n2 - 0.5 * l2_p_m2) * V_dds +
                            nl * (l2_p_m2 - n2) * V_ddp - 0.5 * nl * l2_p_m2 * V_ddd)
    hop_int[8][4] = hop_int[4][8]
    hop_int[8][5] = hop_int[5][8]
    hop_int[8][6] = hop_int[6][8]

    # eg - eg
    hop_int[7][7] = (0.25 * l2_m2**2 * (3 * V_dds - 4 * V_ddp + V_ddd) +
                    l2_p_m2 * V_ddp + n2 * V_ddd)
    hop_int[8][8] = (0.75 * l2_p_m2**2 * V_ddd + 3 * l2_p_m2 * n2 * V_ddp +
                    0.25 * (l2_p_m2 - 2 * n2)**2 * V_dds)
    hop_int[7][8] = sqrt3 * 0.25 * l2_m2 * (n2 * (2 * V_dds - 4 * V_ddp + V_ddd) +
                                           V_ddd - l2_p_m2 * V_dds)
    hop_int[8][7] = hop_int[7][8]

    # =========================================================================
    # d-f interactions (indices 4,5,6,7,8 with 9,10,11,12,13,14,15)
    # Based on Takegahara et al., J. Phys. C 13, 583 (1980)
    # =========================================================================
    # dxy (index 4) - f interactions
    hop_int[4][9] = sqrt(10.0)/4.0 * lm * (5*n2 - 1) * V_dfs - sqrt(10.0)/2.0 * lm * n2 * V_dfp
    hop_int[4][10] = sqrt(30.0)/8.0 * m * (l2*(5*n2-1) - (5*n2-1)/3.0) * V_dfs + sqrt(30.0)/4.0 * m * (n2 - l2*(5*n2-1)/3.0) * V_dfp
    hop_int[4][11] = sqrt(30.0)/8.0 * l * (m2*(5*n2-1) - (5*n2-1)/3.0) * V_dfs + sqrt(30.0)/4.0 * l * (n2 - m2*(5*n2-1)/3.0) * V_dfp
    hop_int[4][12] = sqrt(2.0)/2.0 * lm * n * l2_m2 * V_dfs + sqrt(2.0) * lm * n * (1 - l2_m2/2.0) * V_dfp
    hop_int[4][13] = sqrt(2.0) * n * l2 * m2 * V_dfs - sqrt(2.0) * n * (l2 + m2 - 2*l2*m2) * V_dfp
    hop_int[4][14] = sqrt(6.0)/8.0 * m * (l2*(l2-3*m2)/l2_p_m2 if l2_p_m2 > 1e-10 else 0) * V_dfs - sqrt(6.0)/4.0 * m * l2 * V_dfp if l2_p_m2 > 1e-10 else 0
    hop_int[4][15] = sqrt(6.0)/8.0 * l * (m2*(3*l2-m2)/l2_p_m2 if l2_p_m2 > 1e-10 else 0) * V_dfs - sqrt(6.0)/4.0 * l * m2 * V_dfp if l2_p_m2 > 1e-10 else 0

    # dyz (index 5) - f interactions
    hop_int[5][9] = sqrt(10.0)/4.0 * mn * (5*n2 - 3) * V_dfs + sqrt(10.0)/2.0 * mn * (1 - n2) * V_dfp
    hop_int[5][10] = sqrt(30.0)/8.0 * lmn * (5*n2 - 1) * V_dfs - sqrt(30.0)/4.0 * lmn * (1 + n2) * V_dfp
    hop_int[5][11] = sqrt(30.0)/8.0 * (m2*n*(5*n2-1) - n*(5*n2-1)/3.0) * V_dfs + sqrt(30.0)/4.0 * n * (m2 - (5*n2-1)*m2/3.0) * V_dfp
    hop_int[5][12] = sqrt(2.0)/2.0 * mn * n * l2_m2 * V_dfs - sqrt(2.0) * mn * (l2 - m2/2.0) * V_dfp
    hop_int[5][13] = sqrt(2.0) * l * m2 * n2 * V_dfs - sqrt(2.0) * l * (m2 + n2 - 2*m2*n2) * V_dfp
    hop_int[5][14] = sqrt(6.0)/8.0 * mn * (l2 - 3*m2) * V_dfs - sqrt(6.0)/4.0 * mn * (l2 - m2) * V_dfp
    hop_int[5][15] = sqrt(6.0)/8.0 * nl * (3*l2 - m2) * V_dfs - sqrt(6.0)/4.0 * nl * (l2 - m2) * V_dfp

    # dxz (index 6) - f interactions
    hop_int[6][9] = sqrt(10.0)/4.0 * nl * (5*n2 - 3) * V_dfs + sqrt(10.0)/2.0 * nl * (1 - n2) * V_dfp
    hop_int[6][10] = sqrt(30.0)/8.0 * (l2*n*(5*n2-1) - n*(5*n2-1)/3.0) * V_dfs + sqrt(30.0)/4.0 * n * (l2 - (5*n2-1)*l2/3.0) * V_dfp
    hop_int[6][11] = sqrt(30.0)/8.0 * lmn * (5*n2 - 1) * V_dfs - sqrt(30.0)/4.0 * lmn * (1 + n2) * V_dfp
    hop_int[6][12] = sqrt(2.0)/2.0 * nl * n * l2_m2 * V_dfs + sqrt(2.0) * nl * (m2 - l2/2.0) * V_dfp
    hop_int[6][13] = sqrt(2.0) * m * l2 * n2 * V_dfs - sqrt(2.0) * m * (l2 + n2 - 2*l2*n2) * V_dfp
    hop_int[6][14] = sqrt(6.0)/8.0 * nl * (l2 - 3*m2) * V_dfs + sqrt(6.0)/4.0 * nl * (l2 - m2) * V_dfp
    hop_int[6][15] = sqrt(6.0)/8.0 * mn * (3*l2 - m2) * V_dfs + sqrt(6.0)/4.0 * mn * (l2 - m2) * V_dfp

    # dx2-y2 (index 7) - f interactions
    hop_int[7][9] = sqrt(10.0)/8.0 * n * l2_m2 * (5*n2 - 1) * V_dfs - sqrt(10.0)/4.0 * n * l2_m2 * n2 * V_dfp
    hop_int[7][10] = sqrt(30.0)/16.0 * l * l2_m2 * (5*n2 - 1) * V_dfs - sqrt(30.0)/8.0 * l * (n2*l2_m2 - l2 + m2) * V_dfp
    hop_int[7][11] = sqrt(30.0)/16.0 * m * l2_m2 * (5*n2 - 1) * V_dfs + sqrt(30.0)/8.0 * m * (n2*l2_m2 + l2 - m2) * V_dfp
    hop_int[7][12] = sqrt(2.0)/4.0 * n * l2_m2**2 * V_dfs + sqrt(2.0)/2.0 * n * l2_p_m2 * (1 - l2_m2**2/(l2_p_m2**2 if l2_p_m2 > 1e-10 else 1)) * V_dfp if l2_p_m2 > 1e-10 else 0
    hop_int[7][13] = sqrt(2.0)/2.0 * lm * n * l2_m2 * V_dfs - sqrt(2.0) * lm * n * V_dfp
    hop_int[7][14] = sqrt(6.0)/16.0 * l * l2_m2 * (l2 - 3*m2) / (l2_p_m2 if l2_p_m2 > 1e-10 else 1) * V_dfs if l2_p_m2 > 1e-10 else 0
    hop_int[7][15] = sqrt(6.0)/16.0 * m * l2_m2 * (3*l2 - m2) / (l2_p_m2 if l2_p_m2 > 1e-10 else 1) * V_dfs if l2_p_m2 > 1e-10 else 0

    # dz2 (index 8) - f interactions
    hop_int[8][9] = (n2*(5*n2-3)/2.0 + 3.0/4.0*(1-n2)*(5*n2-1)) * V_dfs + sqrt(6.0)/2.0 * n2*(1-n2) * V_dfp
    hop_int[8][10] = sqrt(2.0)/8.0 * l * ((5*n2-1)*(3*n2-1) - 6*n2) * V_dfs + sqrt(3.0)/2.0 * l * n2 * (1 - n2) * V_dfp
    hop_int[8][11] = sqrt(2.0)/8.0 * m * ((5*n2-1)*(3*n2-1) - 6*n2) * V_dfs + sqrt(3.0)/2.0 * m * n2 * (1 - n2) * V_dfp
    hop_int[8][12] = sqrt(30.0)/8.0 * n * l2_m2 * (n2 - 0.5*l2_p_m2) * V_dfs + sqrt(30.0)/4.0 * n * l2_m2 * (1 - n2) * V_dfp
    hop_int[8][13] = sqrt(30.0)/4.0 * lm * n * (n2 - 0.5*l2_p_m2) * V_dfs - sqrt(30.0)/2.0 * lm * n * (1 - n2) * V_dfp
    hop_int[8][14] = sqrt(10.0)/16.0 * l * (l2 - 3*m2) * (3*n2 - 1) * V_dfs - sqrt(10.0)/8.0 * l * (l2 - 3*m2) * n2 * V_dfp
    hop_int[8][15] = sqrt(10.0)/16.0 * m * (3*l2 - m2) * (3*n2 - 1) * V_dfs - sqrt(10.0)/8.0 * m * (3*l2 - m2) * n2 * V_dfp

    # Transpose (d even, f odd -> opposite sign)
    for i in range(4, 9):
        for j in range(9, 16):
            hop_int[j][i] = -hop_int[i][j]

    # =========================================================================
    # f-f interactions (indices 9-15 with 9-15)
    # Based on Takegahara et al. and symmetry considerations
    # =========================================================================
    # Diagonal terms
    # fz3-fz3
    hop_int[9][9] = (n2 * (5*n2-3)**2/4.0 * V_ffs +
                    3.0/2.0 * n2 * (1-n2) * (5*n2-1) * V_ffp +
                    15.0/4.0 * (1-n2)**2 * V_ffd)

    # fxz2-fxz2
    hop_int[10][10] = (3.0/8.0 * l2 * (5*n2-1)**2 * V_ffs +
                      l2 * (1 - 5*n2/2.0)**2 + n2 * (1 - l2) * V_ffp +
                      (1-l2) * (1-n2) * V_ffd)

    # fyz2-fyz2
    hop_int[11][11] = (3.0/8.0 * m2 * (5*n2-1)**2 * V_ffs +
                      m2 * (1 - 5*n2/2.0)**2 + n2 * (1 - m2) * V_ffp +
                      (1-m2) * (1-n2) * V_ffd)

    # fz(x2-y2)-fz(x2-y2)
    hop_int[12][12] = (15.0/4.0 * n2 * l2_m2**2 * V_ffs +
                      (l2_p_m2 - n2*l2_m2)**2/l2_p_m2 * V_ffp if l2_p_m2 > 1e-10 else 0 +
                      l2*m2 * V_ffd)

    # fxyz-fxyz
    hop_int[13][13] = (15.0 * l2*m2*n2 * V_ffs +
                      (l2*m2 + m2*n2 + n2*l2 - 4*l2*m2*n2) * V_ffp +
                      (l2*m2 + m2*n2 + n2*l2 - l2*m2*n2) * V_ffd)

    # fx(x2-3y2)-fx(x2-3y2)
    hop_int[14][14] = (5.0/8.0 * l2 * (l2-3*m2)**2 / l2_p_m2**2 * V_ffs if l2_p_m2 > 1e-10 else 0 +
                      (1-l2) * (l2-3*m2)**2 / (4*l2_p_m2) * V_ffp if l2_p_m2 > 1e-10 else 0 +
                      m2 * (3*l2+m2)**2 / (4*l2_p_m2) * V_ffd if l2_p_m2 > 1e-10 else 0)

    # fy(3x2-y2)-fy(3x2-y2)
    hop_int[15][15] = (5.0/8.0 * m2 * (3*l2-m2)**2 / l2_p_m2**2 * V_ffs if l2_p_m2 > 1e-10 else 0 +
                      (1-m2) * (3*l2-m2)**2 / (4*l2_p_m2) * V_ffp if l2_p_m2 > 1e-10 else 0 +
                      l2 * (l2+3*m2)**2 / (4*l2_p_m2) * V_ffd if l2_p_m2 > 1e-10 else 0)

    # Off-diagonal f-f terms (selected key terms)
    # fz3-fxz2
    hop_int[9][10] = sqrt(6.0)/8.0 * nl * (5*n2-1) * (5*n2-3) * V_ffs + sqrt(6.0)/4.0 * nl * (5*n2-1) * (1-n2) * V_ffp

    # fz3-fyz2
    hop_int[9][11] = sqrt(6.0)/8.0 * mn * (5*n2-1) * (5*n2-3) * V_ffs + sqrt(6.0)/4.0 * mn * (5*n2-1) * (1-n2) * V_ffp

    # fz3-fz(x2-y2)
    hop_int[9][12] = sqrt(10.0)/8.0 * n2 * l2_m2 * (5*n2-3) * V_ffs + sqrt(10.0)/4.0 * l2_m2 * (1-n2) * (5*n2-1) * V_ffp

    # fz3-fxyz
    hop_int[9][13] = sqrt(10.0)/4.0 * lm * n2 * (5*n2-3) * V_ffs + sqrt(10.0)/2.0 * lm * (1-n2) * (5*n2-1) * V_ffp

    # fz3-fx(x2-3y2)
    hop_int[9][14] = sqrt(30.0)/16.0 * nl * (l2-3*m2) * (5*n2-3) * V_ffs

    # fz3-fy(3x2-y2)
    hop_int[9][15] = sqrt(30.0)/16.0 * mn * (3*l2-m2) * (5*n2-3) * V_ffs

    # fxz2-fyz2
    hop_int[10][11] = 3.0/8.0 * lm * (5*n2-1)**2 * V_ffs + lm * (1-5*n2/2.0)**2 * V_ffp

    # fxz2-fz(x2-y2)
    hop_int[10][12] = sqrt(10.0)/16.0 * l * l2_m2 * (5*n2-1) * V_ffs

    # fxz2-fxyz
    hop_int[10][13] = sqrt(10.0)/8.0 * m * n * l * (5*n2-1) * V_ffs

    # fyz2-fz(x2-y2)
    hop_int[11][12] = sqrt(10.0)/16.0 * m * l2_m2 * (5*n2-1) * V_ffs

    # fyz2-fxyz
    hop_int[11][13] = sqrt(10.0)/8.0 * l * n * m * (5*n2-1) * V_ffs

    # fz(x2-y2)-fxyz
    hop_int[12][13] = sqrt(6.0)/4.0 * lm * n * l2_m2 * V_ffs

    # fxyz-fx(x2-3y2)
    hop_int[13][14] = sqrt(6.0)/8.0 * m * n * (l2-3*m2) * V_ffs

    # fxyz-fy(3x2-y2)
    hop_int[13][15] = sqrt(6.0)/8.0 * l * n * (3*l2-m2) * V_ffs

    # fx(x2-3y2)-fy(3x2-y2)
    hop_int[14][15] = (5.0/8.0 * lm * (l2-3*m2)*(3*l2-m2) / l2_p_m2**2 * V_ffs if l2_p_m2 > 1e-10 else 0)

    # Symmetrize f-f
    for i in range(9, 16):
        for j in range(i + 1, 16):
            hop_int[j][i] = hop_int[i][j]

    # =========================================================================
    # S (excited s) interactions (index 16)
    # =========================================================================
    hop_int[16][16] = V_SSs

    # S-s
    hop_int[0][16] = V_sSs
    hop_int[16][0] = V_sSs

    # S-p
    hop_int[16][1] = l * V_Sps
    hop_int[16][2] = m * V_Sps
    hop_int[16][3] = n * V_Sps
    hop_int[1][16] = -hop_int[16][1]
    hop_int[2][16] = -hop_int[16][2]
    hop_int[3][16] = -hop_int[16][3]

    # S-d
    hop_int[16][4] = sqrt3 * lm * V_Sds
    hop_int[16][5] = sqrt3 * mn * V_Sds
    hop_int[16][6] = sqrt3 * nl * V_Sds
    hop_int[16][7] = sqrt3 / 2.0 * l2_m2 * V_Sds
    hop_int[16][8] = (n2 - 0.5 * l2_p_m2) * V_Sds
    for i in range(4, 9):
        hop_int[i][16] = hop_int[16][i]

    # S-f
    hop_int[16][9] = n * (5*n2 - 3) / 2.0 * V_Sfs
    hop_int[16][10] = sqrt(3.0/8.0) * l * (5*n2 - 1) * V_Sfs
    hop_int[16][11] = sqrt(3.0/8.0) * m * (5*n2 - 1) * V_Sfs
    hop_int[16][12] = sqrt15 / 2.0 * n * l2_m2 * V_Sfs
    hop_int[16][13] = sqrt15 * lmn * V_Sfs
    hop_int[16][14] = sqrt(5.0/8.0) * l * (l2 - 3*m2) * V_Sfs
    hop_int[16][15] = sqrt(5.0/8.0) * m * (3*l2 - m2) * V_Sfs
    for i in range(9, 16):
        hop_int[i][16] = -hop_int[16][i]

    return hop_int


# Orbital index mapping for reference
ORBITAL_INDEX = {
    "s": 0,
    "px": 1, "py": 2, "pz": 3,
    "dxy": 4, "dyz": 5, "dxz": 6, "dx2-y2": 7, "dz2": 8,
    "fz3": 9, "fxz2": 10, "fyz2": 11, "fz(x2-y2)": 12,
    "fxyz": 13, "fx(x2-3y2)": 14, "fy(3x2-y2)": 15,
    "S": 16
}
