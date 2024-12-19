# Below are all the colormap definitions
# that were created for the Krause and Klaus
# paper and subsequent Klaus and Krause paper
# The colormaps might be useful elsewhere.
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import (BoundaryNorm, ListedColormap,
                               LinearSegmentedColormap, Normalize)

# fixme VK: these packages are quite "non-standard";
#  I would recommend defining the colormaps without these packages in the final version
#  see https://github.com/ARM-DOE/pyart/blob/main/pyart/graph/_cm.py
import cmasher as cmr  # fixme VK
import cmocean  # fixme VK


def get_hs_cmap(vmax, updraft_threshold=0.2, extend=True):
    # vmin and vmax need to have the same absolute value

    grey_fraction = updraft_threshold / vmax
    color_fraction = 1 - grey_fraction

    total_N = 1000 / grey_fraction
    color_N = int(color_fraction * total_N / 2)
    grey_N = int(grey_fraction * total_N / 2)

    if extend:
        color_N += 1

    seismic = plt.get_cmap('seismic')
    blue_s = seismic(np.linspace(0, 0.40, color_N))
    red_s = seismic(np.linspace(0.60, 1, color_N))
    #greys = matplotlib.colormaps['Greys']
    greys = plt.get_cmap('Greys')
    grey1 = greys(np.linspace(0.45, 0.55, grey_N))
    grey1_r = greys(np.linspace(0.55, 0.45, grey_N))

    newcolors = np.vstack((blue_s, grey1_r, grey1, red_s))
    hs_cmap = LinearSegmentedColormap.from_list(name='hs_cmap',colors=newcolors)

    # plt.figure(figsize=(10, 2))
    # plt.imshow(np.vstack((np.linspace(0, 1, 256), np.linspace(0, 1, 256))),
    #            aspect='auto', cmap=hs_cmap)
    # plt.subplots
    # _adjust(top=1, bottom=0, right=1, left=0)
    # plt.show()

    return hs_cmap

def get_obj_cmap():
    # Get the existing colormap
    cmap = plt.get_cmap('tab20b')
    # Extract the colors
    colors = cmap(np.linspace(0, 1, cmap.N))
    #
    #we need a colormap to 200 (or more?), but with alternating colors that are not close 
    #
    color_index = [0,19,4,15,8,3,16,7,12,9,1,18,5,14,10,17,2,13,6,11]
    object_colors = []
    white = [1., 1., 1. , 1.]
    object_colors.append(white)
    
    for l in range(10): #need 200 so 20*10
        for n in range(cmap.N):
            object_colors.append(colors[color_index[n]])
            
    object_cmap = LinearSegmentedColormap.from_list(name='object_index', colors=object_colors)
    return object_cmap
    
def get_zdr_cmap_NWS():
    # NWS like Zdr color map
    # Use only -4db to 8db on range
    nodes = [0.000, 0.330, 0.335, 0.354, 0.416, 0.458, 0.500, 0.580, 0.666, 0.750, 0.833, 1.000]
    zdr_colors = [[0, 0, 0],
                  [0.86, 0.86, 0.86],
                  [0.55, 0.47, 0.71],
                  [0.04, 0.04, 0.61],
                  [0.26, 0.97, 0.83],
                  [0.35, 0.86, 0.38],
                  [1.00, 1.00, 0.40],
                  [0.86, 0.04, 0.02],
                  [0.69, 0.00, 0.00],
                  [0.94, 0.47, 0.71],
                  [1.00, 1.00, 1.00],
                  [0.57, 0.18, 0.58]
                  ]
    # From Jacob Carlin who did all the hard work, 10/17/2023, personal communications
    #   zdr_cmap = {0.000: (0.00, 0.00, 0.00),
    #                0.330: (0.86, 0.86, 0.86),
    # 0.335: (0.55, 0.47, 0.71),
    # 0.354: (0.04, 0.04, 0.61),
    # 0.416: (0.26, 0.97, 0.83),
    # 0.458: (0.35, 0.86, 0.38),
    # 0.500: (1.00, 1.00, 0.40),
    # 0.580: (0.86, 0.04, 0.02),
    # 0.666: (0.69, 0.00, 0.00),
    # 0.750: (0.94, 0.47, 0.71),
    # 0.833: (1.00, 1.00, 1.00),
    # 1.000: (0.57, 0.18, 0.58)} #

    # example in python
    # nodes = [0.0, 0.4, 0.8, 1.0]
    # cmap2 = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    zdr_cmap = LinearSegmentedColormap.from_list(name='zdr_cmap', colors=list(zip(nodes, zdr_colors)))

    # plt.figure(figsize=(10, 2))
    # plt.imshow(np.vstack((np.linspace(0, 1, 256), np.linspace(0, 1, 256))),
    #            aspect='auto', cmap=hs_cmap)
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
    # plt.show()

    return zdr_cmap


def get_hs_overlay_cmap():
    clist = [[254, 254, 254, 1], [255, 255, 255, 1]]
    hs_overlay_cmap = LinearSegmentedColormap.from_list(name='hs_overlap_cmap', colors=clist)
    return hs_overlay_cmap


def get_zmask_cmap():
    # nodes = [0.0,1.0]
    # colors = [ [255,255,255,1], [23, 82,126,1] ]
    zmask_cmap = ListedColormap(['grey', 'blue'])
    norm = BoundaryNorm([0, 1], zmask_cmap.N)
    # zmask_cmap = LinearSegmentedColormap.from_list( 'zmask_cmap', list(zip(nodes, colors)))
    return zmask_cmap


def get_altzdr_cmap(vmax, rain_threshold=0.25, extend=True):
    # vmin and vmax need to have the same absolute value

    grey_fraction = rain_threshold / vmax
    color_fraction = 1 - grey_fraction

    total_N = 1000 / grey_fraction
    color_N = int(color_fraction * total_N / 2)
    grey_N = int(grey_fraction * total_N / 2)

    if extend:
        color_N += 1

    winter = plt.get_cmap('winter')
    blue_s = winter(np.linspace(0.5, 0.0, color_N))
    green_s = winter(np.linspace(1, 0.5, color_N))

    greys = plt.get_cmap('Greys')
    grey1 = greys(np.linspace(0.60, 0.40, grey_N))
    grey1_r = greys(np.linspace(0.40, 0.60, grey_N))

    newcolors = np.vstack((blue_s, grey1, grey1_r, green_s))
    zdr_cmap = LinearSegmentedColormap.from_list(name='zdr_cmap', colors=newcolors)

    # plt.figure(figsize=(10, 2))
    # plt.imshow(np.vstack((np.linspace(0, 1, 256), np.linspace(0, 1, 256))),
    #            aspect='auto', cmap=hs_cmap)
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
    # plt.show()
    zdr_cmap.set_over(green_s(1))
    zdr_cmap.set_under(blue_s(0))

    return zdr_cmap


def get_zdr_cmap(vmin, vmax, include_norm=False):
    width = vmax - vmin
    center = abs(vmin / width)
    zdr_cmap = cmocean.cm.diff
    # zdr_cmap.set_over(zdr_cmap(1.0))
    # zdr_cmap.set_under(zdr_cmap(0.0))
    # vmin needs to be < 0, vmax > 0

    zdr_N = 100
    zdr_s = zdr_cmap(np.linspace(0.0, 1.0, zdr_N))

    # colors = [ zdr_s[0], zdr_s[50], zdr_s[99]]
    colors = ['#893584',  # purple
              '#00ffff',  # blue
              '#a6a6a6', '#c0c0c0', '#a6a6a6',  # 2x grey
              '#ffff00',  # dark yellow
              '#cc5200']  # dark orange

    #    ZDR_cmap = LinearSegmentedColormap.from_list('zdr_cmap', zdr_s)
    #    zdr_cmap = ZDR_cmap
    nodes = [0,
             center - 0.5 / width,
             center - 0.25 / width,
             center,
             center + 0.25 / width,
             center + 0.5 / width,
             1]

    zdr_cmap = LinearSegmentedColormap.from_list(
        name="zdr_cmap", colors=list(zip(nodes, colors)))

    if include_norm:
        values = np.linspace(vmin, vmax, int(width / 0.1))
        norm = Normalize(vmin=min(values), vmax=max(values))
        return zdr_cmap, norm
    else:
        return zdr_cmap


def get_zdr_cmap_orig(vmin, vmax, include_norm=False):
    # vmin needs to be < 0, vmax > 0

    colors = ['#cc5200',  # dark orange
              '#893584',  # purple
              '#a6a6a6', '#a6a6a6',  # 2x grey
              '#68ad45',  # green
              '#0066ff']  # blue

    width = vmax - vmin
    center = -vmin / width

    nodes = [0, center / 1.5, center - 0.1 / width, center + 0.1 / width, 0.4, 1]
    zdr_cmap = LinearSegmentedColormap.from_list(
        name="zdr_cmap", colors=list(zip(nodes, colors)))
    zdr_cmap.set_over('#0066ff')
    zdr_cmap.set_under('#cc5200')

    if include_norm:
        values = np.linspace(vmin, vmax, int((width) / 0.1))
        norm = Normalize(vmin=min(values), vmax=max(values))
        return zdr_cmap, norm
    else:
        return zdr_cmap


def get_dr_cmap(vmin, vmax, precip_threshold=-12, extend=True):
    #    DR_cmap = colormaps['Spectral_r']
    #    DR_cmap.set_over(DR_cmap(1.0))
    #    DR_cmap.set_under(DR_cmap(0.0))

    orange_fraction = abs(precip_threshold / (vmax - vmin))
    blue_fraction = 1 - orange_fraction

    orange_N = int(1000 * orange_fraction)
    blue_N = 1000 - orange_N

    if extend:
        orange_N += 1

    blue = plt.get_cmap('Blues')
    blue_s = blue(np.linspace(0.2, 0.6, blue_N))
    orange = plt.get_cmap('Oranges')
    orange_s = orange(np.linspace(0.50, 0.90, orange_N))

    newcolors = np.vstack((blue_s, orange_s))
    DR_cmap = LinearSegmentedColormap.from_list(name='zdr_cmap', colors=newcolors)

    return DR_cmap


def get_NWSRef_ext():
    #
    # This colormap is for the entire scale of the NWS Reflectivity
    # which is -32 to 95 in 0.5 dBZ increments
    #
    nws_cmap = plt.get_cmap('pyart_NWSRef')
    grey = plt.get_cmap('Greys')
    # we want this....
    # vmin = -32
    # vmax = 95
    # we have this......
    vmin = 0
    vmax = 95
    step = 0.5
    color_N = int((vmax - vmin) / step)
    # want to add -32 ->0 or 64 colors from greys
    greys = grey(np.linspace(0.4, 0.8, 64))
    # only from 0dBZ with light blue = 0
    ref_colors = nws_cmap(np.linspace(0, 1, color_N))

    colors = np.vstack((greys, ref_colors))

    ref_cmap = LinearSegmentedColormap.from_list(name='NWSref_ext', colors=colors)
    return ref_cmap


def get_NWSRef_AWIPS():
    # from: https://www.unidata.ucar.edu/blogs/developer/entry/awips-nexrad-level-3-rendered
    #

    colors = ['#D3D3D3',  # light gray 0
              '#9C9C9C',  # medium gray 
              '#767676',  # dark gray 
              '#FFAAAA',  # light pink
              '#EE8C8C',  # medium pink
              '#C97070',  # dark pink
              '#00FB90',  # light green
              '#00BB00',  # medium green
              '#FFFF70',  # light yellow
              '#D0D060',  # dark yellow
              '#FFFF60',  # red
              '#DA0000',  # medium red
              '#AE0000',  # dark red
              '#0000FF',  # blue
              '#FFFFFF',  # white
              '#E700FF',  # purple 75
              '#E700FF']  # purple 75

    nodes = []
    for x in range(0, 80, 5):
        nodes.append(x / 75)

    # print (nodes)

    ref_cmap = LinearSegmentedColormap.from_list(
        name="ref_cmap", colors=list(zip(nodes, colors)))

    ref_cmap.set_over('#E700FF')
    ref_cmap.set_under('#D3D3D3')
    return ref_cmap


def get_NWS_CC_ext():
    #
    # use Vmin = 0.2 and Vmax = 1.05
    # RhoHV or CC in the NWS ranges in 254 values from 0.2083 (coded as 2) to 1.051 (coded as 255) The
    # endocded values of 0 are below threshold (black)
    # encoded values of 1 are range folded (purple haze, not included in cmap)

    rho_main = cmr.get_sub_cmap('pyart_NWSRef', 0.0, 0.80, N=121)
    grey1 = cmr.get_sub_cmap('Greys', 0.3, 0.6, N=119)

    rho_pink = rho_main(1.0)

    rho_white = (0.9, 0.9, 0.9, 1.0)
    # make the top color from pink to white for 14 spots
    rho_top = LinearSegmentedColormap.from_list(
        name="rho_top", colors=[rho_pink, rho_white])

    newcolors = np.vstack((grey1(np.linspace(0, 1, 119)),
                           rho_main(np.linspace(0, 1, 121)),
                           rho_top(np.linspace(0, 1, 14))
                           )
                          )
    # Create colormap normalization from values range
    norm = Normalize(vmin=0.2, vmax=1.05)

    CC_cmap = LinearSegmentedColormap.from_list(name='CC_cmap', colors=newcolors)

    return CC_cmap, norm


def get_cc():
    #
    # use Vmin = 0.6 and Vmax = 1.05
    #
    # A cut down cmap with an expansion of the top colors
    #
    rho_main = cmr.get_sub_cmap('pyart_NWSRef', 0.0, 0.75, N=12)

    # grey1 = cmr.get_sub_cmap('Greys', 0.3, 0.6, N=119)

    # rho_darkred = rho_main(1.0)
    # rho_white = (0.9, 0.9, 0.9, 1.0)  # dirty white for white backgrounds

    # make the top color from pink to white for 14 spots
    # rho_top = LinearSegmentedColormap.from_list(
    #     "rho_top", [rho_darkred, rho_white])

    newcolors = np.vstack((rho_main(np.linspace(0, 1, 12)),))
    cc_cmap = LinearSegmentedColormap.from_list(name='CC_cmap', colors=newcolors)
    cc_cmap.set_under('#D3D3D3')

    norm = Normalize(vmin=0.6, vmax=1.05)

    return cc_cmap, norm


def get_zdrcol_cmap():
    #
    # use Vmin = 0.0 and Vmax = 6.0 for km (or 5km depends)

    ZdrCol_cmap = cmr.get_sub_cmap('terrain', 0.1, 0.5, N=128)

    yellow = ZdrCol_cmap(1.0)
    white = (0.9, 0.9, 0.9, 1.0)
    # make the top color from yellow to white for 16 spots
    zdrcol_top = LinearSegmentedColormap.from_list(
        name="zdrcol_top", colors=[yellow, white])

    newcolors = np.vstack((ZdrCol_cmap(np.linspace(0, 1, 128)),
                           zdrcol_top(np.linspace(0, 1, 16))))

    zdrcol_cmap = LinearSegmentedColormap.from_list(name='ZdrCol_cmap', colors=newcolors)
    return zdrcol_cmap


def get_grcc():
    values = [0.2, 0.45, 0.60, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 1.0,
              1.05]
    colors = [[100, 100, 100],
              [15, 15, 140],
              [10, 10, 190],
              [120, 120, 255],
              [95, 245, 100],
              [135, 215, 10],
              [255, 255, 0],
              [255, 140, 0],
              [225, 3, 0],
              [139, 30, 77],
              [255, 180, 215],
              [164, 54, 150]
              ]
    normed_colors = np.array(colors) / 255

    cc_cmap = LinearSegmentedColormap.from_list(name="cc_cmap", colors=normed_colors, N=len(values))

    # cc_cmap.set_over('#E700FF')
    norm = BoundaryNorm(boundaries=values, ncolors=len(values))
    cc_cmap.set_under('#D3D3D3')
    return cc_cmap, norm


def get_drarsr_cc():
    # use vmin=0.0 and vmax=1.1
    values = [0.0, 0.45, 0.65, 0.75, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99,
              1.0, 1.05, 1.10]
    colors = [[135, 0, 135],
              [149, 149, 156],
              [22, 29, 140],
              [9, 2, 217],
              [137, 135, 214],
              [92, 255, 89],
              [139, 207, 2],
              [255, 196, 0],
              [255, 137, 3],
              [255, 59, 0],
              [227, 0, 0],
              [161, 0, 0],
              [151, 5, 86],
              [250, 172, 209],
              [255, 255, 255]
              ]

    normed_colors = np.array(colors) / 255

    cc_cmap = LinearSegmentedColormap.from_list(
        name="cc_cmap", colors=normed_colors, N=len(values))

    norm = BoundaryNorm(boundaries=values, ncolors=len(values))
    cc_cmap.set_under('#D3D3D3')

    return cc_cmap, norm

if __name__ == '__main__':

    # debugging

    # HotSpot colormap
    hs_cmap = get_hs_cmap(3.0)
    #
    # DR colormap
    # 'Spectral_r', vmin=-20, vmax=0.0
    #
    # Reflectivity Colormap
    # 'pyart_NWSRef' vmin=0, vmax=70.0
    #
    get_zdrcol_cmap()
    #
    # HS_overlay colormap (white only)
    get_hs_overlay_cmap()
