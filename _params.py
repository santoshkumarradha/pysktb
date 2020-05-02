from numpy import sqrt

#[s, px, py, pz, dxy, dyz, dxz, dx2-y2, dz2, S]


def get_hop_int(V_sss=0,
                V_sps=0,
                V_pps=0,
                V_ppp=0,
                V_sds=0,
                V_pds=0,
                V_pdp=0,
                V_dds=0,
                V_ddp=0,
                V_ddd=0,
                V_SSs=0,
                V_sSs=0,
                V_Sps=0,
                V_Sds=0,
                l=0,
                m=0,
                n=0):

    hop_int = [[None for _ in range(10)] for __ in range(10)]

    hop_int[0][0] = V_sss
    hop_int[0][1] = l * V_sps
    hop_int[0][2] = m * V_sps
    hop_int[0][3] = n * V_sps
    hop_int[1][0] = -hop_int[0][1]
    hop_int[2][0] = -hop_int[0][2]
    hop_int[3][0] = -hop_int[0][3]
    hop_int[0][4] = sqrt(3) * l * m * V_sds
    hop_int[0][5] = sqrt(3) * m * n * V_sds
    hop_int[0][6] = sqrt(3) * l * n * V_sds
    hop_int[4][0] = hop_int[0][4]
    hop_int[5][0] = hop_int[0][5]
    hop_int[6][0] = hop_int[0][6]
    hop_int[0][7] = sqrt(3) / 2. * (l**2 - m**2) * V_sds
    hop_int[7][0] = hop_int[0][7]
    hop_int[0][8] = (n**2 - 0.5 * (l**2 + m**2)) * V_sds
    hop_int[8][0] = hop_int[0][8]

    hop_int[1][1] = l**2 * V_pps + (1. - l**2) * V_ppp
    hop_int[1][2] = l * m * (V_pps - V_ppp)
    hop_int[2][1] = hop_int[1][2]
    hop_int[1][3] = l * n * (V_pps - V_ppp)
    hop_int[3][1] = hop_int[1][3]

    hop_int[1][4] = sqrt(3) * l**2 * m * V_pds + m * (1. - 2 * l**2) * V_pdp
    hop_int[1][5] = l * m * n * (sqrt(3) * V_pds - 2 * V_pdp)
    hop_int[1][6] = sqrt(3) * l**2 * n * V_pds + n * (1. - 2 * l**2) * V_pdp
    hop_int[4][1] = -hop_int[1][4]
    hop_int[5][1] = -hop_int[1][5]
    hop_int[6][1] = -hop_int[1][6]

    hop_int[1][7] = 0.5 * sqrt(3) * l * (l**2 - m**2) * V_pds + l * (
        1. - l**2 + m**2) * V_pdp
    hop_int[1][8] = l * (n**2 - 0.5 *
                         (l**2 + m**2)) * V_pds - sqrt(3) * l * n**2 * V_pdp
    hop_int[7][1] = -hop_int[1][7]
    hop_int[8][1] = -hop_int[1][8]

    hop_int[2][2] = m**2 * V_pps + (1. - m**2) * V_ppp
    hop_int[2][3] = m * n * (V_pps - V_ppp)
    hop_int[3][2] = hop_int[2][3]

    hop_int[2][4] = sqrt(3) * m**2 * l * V_pds + l * (1. - 2 * m**2) * V_pdp
    hop_int[2][5] = sqrt(3) * m**2 * n * V_pds + n * (1. - 2 * m**2) * V_pdp
    hop_int[2][6] = l * m * n * (sqrt(3) * V_pds - 2 * V_pdp)
    hop_int[4][2] = -hop_int[2][4]
    hop_int[5][2] = -hop_int[2][5]
    hop_int[6][2] = -hop_int[2][6]

    hop_int[2][7] = 0.5 * sqrt(3) * m * (l**2 - m**2) * V_pds - m * (
        1. + l**2 - m**2) * V_pdp
    hop_int[2][8] = m * (n**2 - 0.5 *
                         (l**2 + m**2)) * V_pds - sqrt(3) * m * n**2 * V_pdp
    hop_int[7][2] = -hop_int[2][7]
    hop_int[8][2] = -hop_int[2][8]

    hop_int[3][3] = n**2 * V_pps + (1. - n**2) * V_ppp

    hop_int[3][4] = l * m * n * (sqrt(3) * V_pds - 2 * V_pdp)
    hop_int[3][5] = sqrt(3) * n**2 * m * V_pds + m * (1. - 2 * n**2) * V_pdp
    hop_int[3][6] = sqrt(3) * n**2 * l * V_pds + l * (1. - 2 * n**2) * V_pdp
    hop_int[4][3] = -hop_int[3][4]
    hop_int[5][3] = -hop_int[3][5]
    hop_int[6][3] = -hop_int[3][6]

    hop_int[3][7] = 0.5 * sqrt(3) * n * (l**2 - m**2) * V_pds - n * (
        l**2 - m**2) * V_pdp
    hop_int[3][8] = n * (n**2 - 0.5 *
                         (l**2 + m**2)) * V_pds + sqrt(3) * n * (l**2 +
                                                                 m**2) * V_pdp
    hop_int[7][3] = -hop_int[3][7]
    hop_int[8][3] = -hop_int[3][8]

    hop_int[4][4] = l**2 * m**2 * (3 * V_dds - 4 * V_ddp +
                                   V_ddd) + (l**2 + m**2) * V_ddp + n**2 * V_ddd
    hop_int[5][5] = m**2 * n**2 * (3 * V_dds - 4 * V_ddp +
                                   V_ddd) + (m**2 + n**2) * V_ddp + l**2 * V_ddd
    hop_int[6][6] = n**2 * l**2 * (3 * V_dds - 4 * V_ddp +
                                   V_ddd) + (n**2 + l**2) * V_ddp + m**2 * V_ddd

    hop_int[4][5] = l * m**2 * n * (3 * V_dds - 4 * V_ddp +
                                    V_ddd) + l * n * (V_ddp - V_ddd)
    hop_int[4][6] = n * l**2 * m * (3 * V_dds - 4 * V_ddp +
                                    V_ddd) + n * m * (V_ddp - V_ddd)
    hop_int[5][6] = m * n**2 * l * (3 * V_dds - 4 * V_ddp +
                                    V_ddd) + m * l * (V_ddp - V_ddd)
    hop_int[5][4] = hop_int[4][5]
    hop_int[6][4] = hop_int[4][6]
    hop_int[6][5] = hop_int[5][6]

    hop_int[4][7] = 0.5 * l * m * (l**2 - m**2) * (3 * V_dds - 4 * V_ddp +
                                                   V_ddd)
    hop_int[5][7] = 0.5 * m * n * ((l**2 - m**2) *
                                   (3 * V_dds - 4 * V_ddp + V_ddd) - 2 *
                                   (V_ddp - V_ddd))
    hop_int[6][7] = 0.5 * n * l * ((l**2 - m**2) *
                                   (3 * V_dds - 4 * V_ddp + V_ddd) + 2 *
                                   (V_ddp - V_ddd))

    hop_int[7][4] = hop_int[4][7]
    hop_int[7][5] = hop_int[5][7]
    hop_int[7][6] = hop_int[6][7]

    hop_int[4][8] = sqrt(3) * (l * m * (n**2 - 0.5 * (l**2 + m**2)) * V_dds -
                               2 * l * m * n**2 * V_ddp + 0.5 * l * m *
                               (1. + n**2) * V_ddd)
    hop_int[5][8] = sqrt(3) * (m * n * (n**2 - 0.5 *
                                        (l**2 + m**2)) * V_dds + m * n *
                               (l**2 + m**2 - n**2) * V_ddp - 0.5 * m * n *
                               (l**2 + m**2) * V_ddd)
    hop_int[6][8] = sqrt(3) * (n * l * (n**2 - 0.5 *
                                        (l**2 + m**2)) * V_dds + n * l *
                               (l**2 + m**2 - n**2) * V_ddp - 0.5 * n * l *
                               (l**2 + m**2) * V_ddd)
    hop_int[8][4] = hop_int[4][8]
    hop_int[8][5] = hop_int[5][8]
    hop_int[8][6] = hop_int[6][8]

    hop_int[7][7] = 0.25 * (l**2 - m**2)**2 * (
        3 * V_dds - 4 * V_ddp + V_ddd) + (l**2 + m**2) * V_ddp + n**2 * V_ddd
    hop_int[8][8] = 0.75 * (l**2 + m**2)**2 * V_ddd + 3 * (
        l**2 + m**2) * n**2 * V_ddp + 0.25 * (l**2 + m**2 - 2 * n**2)**2 * V_dds
    hop_int[7][8] = sqrt(3) * 0.25 * (l**2 - m**2) * (
        n**2 * (2 * V_dds - 4 * V_ddp + V_ddd) + V_ddd - (l**2 + m**2) * V_dds)
    hop_int[8][7] = sqrt(3) * 0.25 * (l**2 - m**2) * (
        n**2 * (2 * V_dds - 4 * V_ddp + V_ddd) + V_ddd - (l**2 + m**2) * V_dds)

    hop_int[9][9] = V_SSs
    hop_int[0][9] = V_sSs
    hop_int[9][0] = V_sSs
    hop_int[9][1] = l * V_Sps
    hop_int[9][2] = m * V_Sps
    hop_int[9][3] = n * V_Sps
    hop_int[1][9] = -hop_int[9][1]
    hop_int[2][9] = -hop_int[9][2]
    hop_int[3][9] = -hop_int[9][3]
    hop_int[9][4] = sqrt(3) * l * m * V_Sds
    hop_int[9][5] = sqrt(3) * m * n * V_Sds
    hop_int[9][6] = sqrt(3) * l * n * V_Sds
    hop_int[4][9] = hop_int[9][4]
    hop_int[5][9] = hop_int[9][5]
    hop_int[6][9] = hop_int[9][6]
    hop_int[9][7] = sqrt(3) / 2. * (l * l - m * m) * V_Sds
    hop_int[7][9] = hop_int[9][7]
    hop_int[9][8] = (n**2 - 0.5 * (l**2 + m**2)) * V_Sds
    hop_int[8][9] = hop_int[9][8]
    return hop_int
