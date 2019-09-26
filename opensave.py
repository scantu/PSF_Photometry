import pandas as pd
import numpy as np


def opendf(filename, sep='\s+', head=0, icol=False):
    # filename=str(filename)
    df = pd.read_csv(filename, sep=sep, header=head, index_col=icol)
    return df


def savedf(df, filename, sep='\t', index=False):
    df.to_csv(filename, sep=sep, index=index)
    return


def idmerge(rdf, gdf, how='inner', on='id'):
    df = pd.merge(rdf, gdf, how=how, on=on)
    return df


def sigd(a, low, high):
    med = abs(a - np.median(a))
    m1 = np.median(med)
    x = m1 * 1.4826
    l = np.median(a) - (low * x)
    h = np.median(a) + (high * x)
    return l, h


def ellipse(x, y, ellip, r_eff, h=0, k=0, pos=0):

    a, b = r_eff, (1 - ellip) * r_eff
    mask = (((x - h) * np.cos(pos) + (y - k) * np.sin(pos))**2 / a**2) + \
        (((x - h) * np.sin(pos) - (y - k) * np.cos(pos))**2 / b**2) <= 1
    return mask


def density_prof(filename, ellip, r_max, asize, h=0, k=0, pos=0):
    print('hi')
    df = opendf(filename)
    x = df.ra_g
    y = df.dec_g
    print(asize, r_max)
    r_eff = np.arange(asize, r_max + asize, asize)
    print(len(r_eff))

    outer_rad = None
    inner_rad = None
    print('broken?')
    prof = np.zeros(len(r_eff))
    print('here')

    a_outer, b_outer = r_eff[0], (1 - ellip) * r_eff[0]
    print('a')
    outer_rad = ((((x - h) * np.cos(pos) + (y - k) * np.sin(pos))**2 /
                  a_outer**2) + (((x - h) * np.sin(pos) - (y - k) *
                                  np.cos(pos))**2 / b_outer**2) <= 1)
    print('b')
    new_df = None
    new_df = df[(outer_rad)]
    print('c')
    prof[0] = len(new_df)

    for i in range(len(r_eff) - 1):
        new_df = 0
        # if r_eff[i] == asize:
        #     a_outer, b_outer = r_eff[i], (1 - ellip) * r_eff[i]

        #     outer_rad = ((((x - h) * np.cos(pos) + (y - k) * np.sin(pos))**2 /
        #                   a_outer**2) + (((x - h) * np.sin(pos) - (y - k) *
        #                                   np.cos(pos))**2 / b_outer**2) <= 1)

        #     new_df = df[(outer_rad)]

        #     prof[i] = len(new_df)

        # else:

        a_outer, b_outer = r_eff[i + 1], (1 - ellip) * r_eff[i + 1]
        a_inner, b_inner = r_eff[i + 1] - \
            asize, (1 - ellip) * (r_eff[i + 1] - asize)

        outer_rad = ((((x - h) * np.cos(pos) + (y - k) * np.sin(pos))**2 / a_outer**2) +
                     (((x - h) * np.sin(pos) - (y - k) * np.cos(pos))**2 / b_outer**2) <= 1)

        inner_rad = ((((x - h) * np.cos(pos) + (y - k) * np.sin(pos))**2 / a_inner**2) +
                     (((x - h) * np.sin(pos) - (y - k) * np.cos(pos))**2 / b_inner**2) > 1)

        new_df = df[(outer_rad) & (inner_rad)]

        prof[i + 1] = len(new_df)

    return prof, r_eff
