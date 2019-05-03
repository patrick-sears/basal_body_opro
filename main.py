#!/usr/bin/python3

import re
import sys
import math
import os

from matplotlib import pyplot as plt

import prs_basal_body_opro_lib as basbo



#################################
# 8.5x11 with 10mm margins, portrait:  7.71 x 10.21 inch
ufigsize = [10.21, 7.71]
udpi = 300.0
#################################

# exec( open('g01_acc.py').read() )

cell_basal_body_color = []
n_cell_basal_body_color = 0
cell_basal_body_color_default = '#aa0000'
cell_basal_body_color_notcell  = '#dddddd'


if sys.argv[1] == '--version':
  progd = os.path.dirname(os.path.realpath(__file__))
  f = open( progd+'/version' )
  for l in f:
    l = l.rstrip()
    print(l)
  exit(0)

############################################
f = open(sys.argv[1])
for l in f:
  if not l.startswith('!'):  continue
  l = l.strip()
  ll = l.split(' ')
  key = ll[0]
  #
  if key == '!00--config_id': config_id = ll[1]
  #
  elif key == '!fname_bbodies':      fname_bbodies    = ll[1]
  elif key == '!fname_boundaries':   fname_boundaries = ll[1]
  elif key == '!fname_zgr1':   fname_zgr1 = ll[1]
  elif key == '!fname_zgr2':   fname_zgr2 = ll[1]
  elif key == '!fname_z3':   fname_z3 = ll[1]
  elif key == '!fname_z4':   fname_z4 = ll[1]
  #
  elif key == '!im_wh_px':
    im_w_px = int(ll[1])
    im_h_px = int(ll[2])
  #
  elif key == '!indata_bbodies_invy': indata_bbodies_invy = int(ll[1])
  elif key == '!indata_boundaries_invy': indata_boundaries_invy = int(ll[1])
  elif key == '!indata_bbodies_units': indata_bbodies_units = ll[1]
  elif key == '!indata_boundaries_units': indata_boundaries_units = ll[1]
  elif key == '!indata_label_body':   indata_label_body = ll[1]
  elif key == '!indata_label_foot':   indata_label_foot = ll[1]
  #
  elif key == '!max_separation_um': max_separation_um = float(ll[1])
  elif key == '!pixel_size_um': pixel_size_um = float(ll[1])
  #
  elif key == '!zgr1_vec_length': zgr1_vec_length = float(ll[1])
  elif key == '!zgr1_body_color':  zgr1_body_color = ll[1]
  elif key == '!zgr1_foot_color':  zgr1_foot_color = ll[1]
  #
  elif key == '!cell_basal_body_color_default':
    ll = l.split(' ')
    cell_basal_body_color_default = '#'+ll[1]
  elif key == '!cell_basal_body_color_notcell':
    ll = l.split(' ')
    cell_basal_body_color_notcell = '#'+ll[1]
  elif key == '!cell_basal_body_color':
    for l in f:
      l = l.strip()
      if len(l) == 0:  break
      if l[0] == '#':  continue
      ll = l.split(' ')
      cell_basal_body_color.append( '#'+ll[0] )
  #
f.close()
############################################

n_cell_basal_body_color = len( cell_basal_body_color )

### for i in range(n_cell_basal_body_color):
###   print("> ", cell_basal_body_color[i])
### exit(0)


im_w_um = im_w_px * pixel_size_um
im_h_um = im_h_px * pixel_size_um



bab = [] # basal body
baf = [] # basal foot

# ba body, fa foot
xba = []
yba = []
xfa = []
yfa = []


############################################
# Load basal body data
f = open(fname_bbodies)
f.readline()  # remove header
for l in f:
  l = l.strip()
  # First find the delimiter.
  # I think re.sub(' +', ' ')  works like this:
  # ' +' means look any number of ' '
  # ' ' means, replace with ' '
  # l = re.sub(' +', ' ', l)
  l = re.sub(r'\s+', ' ', l)  # all white space replaced by a single ' '
  ll = l.split(' ')
  if len(ll) != 3:
    print("Error:  ll != 3.")
    print("  ", l)
    exit(1)
  if ll[2] == indata_label_body:
    xba.append( float(ll[0]) )
    yba.append( float(ll[1]) )
    bab.append( basbo.vec2(float(ll[0]), float(ll[1])) )
    # bab.append( basbo.vec2.from_str_xy(ll[0], ll[1]) )
  elif ll[2] == indata_label_foot:
    xfa.append( float(ll[0]) )
    yfa.append( float(ll[1]) )
    baf.append( basbo.vec2(float(ll[0]), float(ll[1])) )
  else:
    print("Error:  Unexpected ll[2].")
    print("  ll[2]:  ", ll[2])
    exit(1)
f.close()
############################################

n_ba = len(xba)
n_fa = len(xfa)

n_bab = len(bab)
n_baf = len(baf)



cellx = []
celly = []
cell_n = []



pcell = []
n_pcell = 0



############################################
# Load cell periphery data
i = -1
f = open(fname_boundaries)
f.readline()  # remove header
for l in f:
  l = l.rstrip()
  if len(l) == 0:
    i += 1
    cellx.append([])
    celly.append([])
    cell_n.append(0)
    pcell.append( basbo.pogocell() )
    continue
  # ll = l.split('\t')
  l = re.sub(r'\s+', ' ', l)  # all white space replaced by a single ' '
  ll = l.split(' ')
  #
  cellx[i].append( float(ll[0]) )
  celly[i].append( float(ll[1]) )
  cell_n[i] += 1
  #
  pcell[i].add_point_from_str(ll[0], ll[1])
f.close()
############################################
n_cell = len(cellx)
n_cell_plus = n_cell + 1
# n_cell_plus includes the area outside all cells.

n_pcell = len(pcell)






#################################
if indata_bbodies_units == 'px':
  for i in range(n_ba):
    xba[i] *= pixel_size_um
    yba[i] *= pixel_size_um
    bab[i].multiply_scalar( pixel_size_um )
  for i in range(n_fa):
    xfa[i] *= pixel_size_um
    yfa[i] *= pixel_size_um
    baf[i].multiply_scalar( pixel_size_um )
if indata_bbodies_invy == 1:
  for i in range(n_ba):
    yba[i] = im_h_um - yba[i]
    bab[i].y = im_h_um - bab[i].y
  for i in range(n_fa):
    yfa[i] = im_h_um - yfa[i]
    baf[i].y = im_h_um - baf[i].y
#################################



#################################
if indata_boundaries_units == 'px':
  for ci in range(n_cell):
    for pi in range(cell_n[ci]):
      cellx[ci][pi] *= pixel_size_um
      celly[ci][pi] *= pixel_size_um
if indata_boundaries_invy == 1:
  for ci in range(n_cell):
    for pi in range(cell_n[ci]):
      celly[ci][pi] = im_h_um - celly[ci][pi]
#################################
if indata_boundaries_units == 'px':
  for ci in range(n_pcell):
    pcell[ci].rescale( pixel_size_um )
if indata_boundaries_invy == 1:
  for ci in range(n_pcell):
    pcell[ci].apply_invy( im_h_um )
#################################





#################################
# Calculate the centroid of points for each cell.
# Note, this is _not_ the centroid of the plane figure.
cell_cenpx = []
cell_cenpy = []
for ci in range(n_cell):
  cell_cenpx.append(0.0)
  cell_cenpy.append(0.0)
  #
  cell_cenpx[ci], cell_cenpy[ci] = basbo.get_centroid_of_points(cellx[ci],celly[ci])
#################################


#################################
# Sort cell periphery points.
for ci in range(n_cell):
  cellx[ci], celly[ci] = basbo.order_right_handed( cellx[ci], celly[ci], cell_cenpx[ci], cell_cenpy[ci] )
#################################
for ci in range(n_pcell):
  pcell[ci].sort_points()
  pcell[ci].calc_fcent()   # also calculates area
#################################



############################################
# Make closed contour cell polygons for graphing.
# cellx_closed = basbo.create_closed_contour_arrays( cellx )
# celly_closed = basbo.create_closed_contour_arrays( celly )
############################################
cellx_closed = []
celly_closed = []
for ci in range(n_pcell):
  vx, vy = pcell[ci].get_closed_countour_arrays()
  cellx_closed.append( vx )
  celly_closed.append( vy )
############################################

print("n_cell  = ", n_cell)
print("n_pcell = ", n_pcell)




ba_ib = []
babfi = []


#################################
# basal bodies that have a matching foot.
xc0 = []
yc0 = []

# matching foot.
xc1 = []
yc1 = []

bafo = []  # basal body with a matching foot.
n_bafo = 0 # total number

n_bf = 0  # Total number of basal bodies with a matching foot.
#################################


mag2_maxmax = (max_separation_um * 2.0)**2
############################################
for i in range(n_ba):
  print('.', end='', flush=True)
  xb = xba[i]
  yb = yba[i]
  ba_ib.append( -1 )
  babfi.append( -1 )
  #
  mag2_best = mag2_maxmax
  j_best = -1
  ################
  for j in range(n_fa):
    dx = xfa[j] - xb
    dy = yfa[j] - yb
    #
    # First check to see if the foot is in our max sep box.
    if abs(dx) > max_separation_um:  continue
    if abs(dy) > max_separation_um:  continue
    mag2 = dx**2 + dy**2
    if mag2 < mag2_best:
      mag2_best = mag2
      j_best = j
  ################
  #
  ba_ib[i] = j_best
  babfi[i] = j_best
  if j_best != -1:
    n_bf += 1
    xc0.append( xba[i] )
    yc0.append( yba[i] )
    xc1.append( xfa[j_best] )
    yc1.append( yfa[j_best] )
    #
    # bafo.append( bab[i], baf[j_best] )
    bafo.append( basbo.bodyfoot() )
    bafo[n_bafo].set_vec2( bab[i], baf[j_best] )
    n_bafo += 1
    #
print()
############################################



# unit vectors for each base/foot.
xcdu = []
ycdu = []
for i in range(n_bf):
  #
  xcdu.append(0.0)
  ycdu.append(0.0)
  xcdu[i], ycdu[i] = basbo.get_dv_unit( xc0[i], yc0[i], xc1[i], yc1[i] )


# Mean base/foot vector.
mean_xc = 0.0
mean_yc = 0.0
for i in range(n_bf):
  mean_xc += xcdu[i]
  mean_yc += ycdu[i]
mean_xc /= n_bf
mean_yc /= n_bf
mean_c_mag = math.sqrt( mean_xc**2 + mean_yc**2 )


mean_bafo_vec = basbo.vec2()
u = basbo.vec2()
for i in range(n_bafo):
  u = bafo[i].get_uvec()
  mean_bafo_vec.add_vec2( u )





# For graphing.
# x0[0] x1[0] nan x0[1] x1[1] nan...
# y0[0] y1[0] nan y0[1] y1[1] nan...
# gr2_xc, gr2_yc = basbo.get_list_for_vec_graphing( xc0, yc0, xc1, yc1, zgr1_vec_length )

gr2_xc, gr2_yc = basbo.get_list_for_vec_graphing_3(bafo, zgr1_vec_length)


# Check for which cell each point is in.
incell = []
for i in range(n_bf):  incell.append(-1)

# new body->foots are in bafo[]
for i in range(n_bf):
  #######
  # for ci in range(n_cell):
  #   if basbo.in_polygon(cellx[ci], celly[ci], cell_cenpx[ci], cell_cenpy[ci], xc0[i], yc0[i]):
  #     incell[i] = ci
  #     break # stop searching after first cell found.
  #######
  for ci in range(n_cell):
    if pcell[ci].in_cell( bafo[i].body ):
      incell[i] = ci
      break # stop searching after first cell found.
  #######



# Make a different set of points for each cell.
# cex0[0][],cey0[0][] are the points in cell 0.
# cex0[1][],cey0[1][] are the points in cell 1.
# cex0[2][],cey0[2][] are the points in cell 2.
# cex0[3][],cey0[3][] are the points in cell 3.
# ...
# cex0 and cey0 are labeled with a '0' because they
# mark the basal body as opposed to the foot.
### n_cell_plus = n_cell + 1
cex0 = []
cey0 = []

# cex1 is for the basal foot associated with cex0
cex1 = []
cey1 = []

# Unit vectors for each cell.
ce_ux = []
ce_uy = []

for i_ce in range(n_cell_plus):
  cex0.append([])
  cey0.append([])
  cex1.append([])
  cey1.append([])
  ce_ux.append([])
  ce_uy.append([])

for i in range(n_bf):
  i_ce = incell[i]
  if i_ce == -1:  i_ce = n_cell_plus-1
  #
  cex0[i_ce].append( xc0[i] )
  cey0[i_ce].append( yc0[i] )
  #
  cex1[i_ce].append( xc1[i] )
  cey1[i_ce].append( yc1[i] )
  #
  mag = math.sqrt( (xc1[i] - xc0[i])**2 + (yc1[i] - yc0[i])**2 )
  if mag > 0.0:
    dx = (xc1[i] - xc0[i]) / mag
    dy = (yc1[i] - yc0[i]) / mag
    #
    ce_ux[i_ce].append( dx )
    ce_uy[i_ce].append( dy )
  else:
    ce_ux[i_ce].append( 0.0 )
    ce_uy[i_ce].append( 0.0 )







############################################
# Get polygon stats for each cell.
cell_area = []
cell_cenfx = []  # center of figure (plane figure)
cell_cenfy = []

for ci in range(n_cell):
  area, fx, fy = basbo.polygon_params(cellx[ci], celly[ci])
  #
  cell_area.append(area)
  cell_cenfx.append(fx)
  cell_cenfy.append(fy)
  print("_cell[",ci,"]  area = {0:0.1f} um^2".format( cell_area[ci] ) )


for ci in range(n_cell):
  print("pcell[",ci,"]  area = {0:0.1f} um^2".format( pcell[ci].area ) )


#######################################################
# *** For each cell, find the mean vector.
#
# Data to use...
# Cell i...
#    Has border:            cellx_closed[i] celly_closed[i]
#    has basal bodies at:   cex0[i][], cey0[i][]
#
#
# Full list of basal bodies that have a matching foot
# is stored in xc0[] yc0[]
# So basal body j has coordinates xc0[j] yc0[j]
# A basal body is in cell i if incell[j] = i.
#
# Note that this includes basal bodies not in any
# cell.  They are listed as belonging to cell -1,
# ie incell[j] = -1.
# 
# The basal foot matching xc0[j] yc0[j] is xc1[j] yc1[j].
# The unit vector fro c0 to c1 is fot that basal body
# to basal foot is  xcdu[j] ycdu[j]
#
# The number of cells is n_cell.
# n_cell_plus  is n_cell + 1 and includes basal bodies not in any cell.
#
#

# Mean of unit vectors for each cell including the
# virtual "no cell".
#################################
cell_mean_vec_x = []
cell_mean_vec_y = []
cell_mean_vec_mag = []
for i in range(n_cell_plus):
  cell_mean_vec_x.append(0.0)
  cell_mean_vec_y.append(0.0)
  cell_mean_vec_mag.append(0.0)
for i in range(n_cell_plus):
  n = len( cex0[i] )
  for j in range( n ):
    cell_mean_vec_x[i] += ce_ux[i][j]
    cell_mean_vec_y[i] += ce_uy[i][j]
  #
  cell_mean_vec_x[i] /= n
  cell_mean_vec_y[i] /= n
  cell_mean_vec_mag[i] = math.sqrt( (cell_mean_vec_x[i]**2) + (cell_mean_vec_y[i]**2) )
#################################





############################################
############################################
gr_cell_uvec_r = 3.0
circ = []
p0 = basbo.vec2()
for i in range(n_cell):
  # p0.set_xy( cell_cenfx[i], cell_cenfy[i] )
  p0.set_xy( pcell[i].fcent.x, pcell[i].fcent.y )
  circ.append( basbo.circle() )
  circ[i].set(40, p0, gr_cell_uvec_r)




gr_circ_x = []
gr_circ_y = []
for i in range(n_cell):
  xarray, yarray = circ[i].get_x_y_arrays()
  gr_circ_x.append( xarray )
  gr_circ_y.append( yarray )


# HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH HERE
# - Is there a cell_mean_vec... for pcell objects?
gr_cell_mean_vx, gr_cell_mean_vy = basbo.get_list_for_vec_graphing_2(
  cell_cenfx, cell_cenfy,
  cell_mean_vec_x,
  cell_mean_vec_y,
  gr_cell_uvec_r,
  )








#######################################################
#######################################################
#######################################################





############################################
fz = open('z02_sum.data', 'w')
line = '\n'
line += '\n'
line += 'n_ba {0:0d}\n'.format(n_ba)
line += 'n_fa {0:0d}\n'.format(n_fa)
line += 'n_bf {0:0d}\n'.format(n_bf)
line += 'n_cell {0:0d}\n'.format(n_cell)
line += '\n'
line += 'mean_[xc,yc] {0:0.6f} {1:0.6f}\n'.format(mean_xc, mean_yc)
line += 'mean_c_mag {0:0.6f}\n'.format(mean_c_mag)
line += '\n'
line += '----------------------------------------------\n'
line += 'mean of unit vectors, basal body -> basal foot\n'
line += '------------\n'
line += 'cell_id dx dy mag\n'
for i in range(n_cell_plus):
  line += 'cell['+str(i)+']'
  line += ' {0:0.6f}'.format( cell_mean_vec_x[i] )
  line += ' {0:0.6f}'.format( cell_mean_vec_y[i] )
  line += ' {0:0.6f}'.format( cell_mean_vec_mag[i] )
  line += '\n'
line += '------------\n'
line += 'Last cell is actually for basal bodies in no cell.\n'
line += '----------------------------------------------\n'
line += '\n'
fz.write(line)
fz.close()
############################################



############################################
fz = open('z01a_plot.data', 'w')
for i in range(n_bf):
  line = str(i)
  line += ' {0:0.6f} {1:0.6f}'.format(xc0[i], yc0[i])
  fz.write(line+'\n')
fz.close()
############################################

############################################
fz = open('z01b_plot.data', 'w')
for i in range(n_bf):
  x1 = xc0[i] + (xc1[i]-xc0[i]) * zgr1_vec_length
  y1 = yc0[i] + (yc1[i]-yc0[i]) * zgr1_vec_length
  line = str(i)
  line += ' {0:0.6f} {1:0.6f}\n'.format(xc0[i], yc0[i])
  fz.write(line)
  line = str(i)
  line += ' {0:0.6f} {1:0.6f}\n'.format(x1, y1)
  fz.write(line)
  line = '- - -\n'
  fz.write(line)
fz.close()
############################################











#######################################################
#######################################################
# Graphing
gr1_markersize = 2.8


############################################
# Useful for testing while working over ssh.
if not 'DISPLAY' in os.environ:
  print( "Didn't find DISPLAY environment variable." )
  print( "  It's needed for matplotlib so exiting" )
  print( "  before creating graphs.")
  exit(0)
############################################



#######################################################
plt.clf()

fig, ax = plt.subplots(figsize=ufigsize, dpi=udpi)
# ax.plot( xba, yba, 'o', markeredgecolor='#ff0000', markerfacecolor='#00000000' )
# ax.plot( xfa, yfa, 'o', markeredgecolor='#00ff00', markerfacecolor='#00000000' )
ax.plot( xba, yba, 'o', markeredgecolor=zgr1_body_color, markerfacecolor='#00000000', markersize=gr1_markersize )
ax.plot( xfa, yfa, 'o', markeredgecolor=zgr1_foot_color, markerfacecolor='#00000000', markersize=gr1_markersize )

ax.plot( gr2_xc, gr2_yc, '-', color='#000055')

for ci in range(n_cell):
  ax.plot( cellx_closed[ci], celly_closed[ci], '-', color='#777777' )


# plt.axis('equal')
ax.set_aspect('equal')
ax.set_xlim( 0.0, im_w_um )
ax.set_ylim( 0.0, im_h_um )
plt.xlabel("μm", fontsize=20)
plt.ylabel("μm", fontsize=20)

plt.savefig(fname_zgr1, dpi=udpi)



#######################################################
plt.clf()

fig, ax = plt.subplots(figsize=ufigsize, dpi=udpi)
ax.plot( xba, yba, 'o', markeredgecolor=zgr1_body_color, markerfacecolor='#00000000' )
ax.plot( xfa, yfa, 'o', markeredgecolor=zgr1_foot_color, markerfacecolor='#00000000' )

ax.plot( gr2_xc, gr2_yc, '-', color='#000055')

for ci in range(n_cell):
  ax.plot( cellx_closed[ci], celly_closed[ci], '-', color='#777777' )

# plt.axis('equal')
ax.set_aspect('equal')
ax.set_xlim( 12.0, 17.0 )
ax.set_ylim( 10.0, 14.0 )

plt.xlabel("μm", fontsize=20)
plt.ylabel("μm", fontsize=20)

plt.savefig(fname_zgr2, dpi=udpi)




#######################################################
plt.clf()

fig, ax = plt.subplots(figsize=ufigsize, dpi=udpi)
# ax.plot( xba, yba, 'o', markeredgecolor='#00ff00', markerfacecolor='#00000000' )
# ax.plot( xfa, yfa, 'o', markeredgecolor='#ff0000', markerfacecolor='#00000000' )
# ax.plot( gr2_xc, gr2_yc, '-', color='#000055')

for ci in range(n_cell):
  ax.plot( cellx_closed[ci], celly_closed[ci], '-', color='#777777' )

print("n_cell_plus = ", n_cell_plus)
# seven cells + outside = 8
### ce_color = [
###   '#4400cc',  # dark violet
###   '#0000dd',  # dark blue
###   '#0099ff',  # bright blue green
###   '#00ff00',  # green
###   '#ccff00',  # orange
###   '#ff0000',  # red
###   '#aa0000',  # dark red
###   '#dddddd',  # gray, not in any cell
###   ]

for i_ce in range(n_cell_plus):
  if i_ce == n_cell:
    ce_color = cell_basal_body_color_notcell
  elif i_ce >= n_cell_basal_body_color:
    ce_color = cell_basal_body_color_default
  else:
    ce_color = cell_basal_body_color[i_ce]
  #
  #  ax.plot(
  #    cex0[i_ce], cey0[i_ce], 'o',
  #    markeredgecolor=ce_color[i_ce],
  #    markerfacecolor=ce_color[i_ce],
  #    markersize=gr1_markersize,
  #    )
  ax.plot(
    cex0[i_ce], cey0[i_ce], 'o',
    markeredgecolor=ce_color,
    markerfacecolor=ce_color,
    markersize=gr1_markersize,
    )



# plt.axis('equal')
ax.set_aspect('equal')
ax.set_xlim( 0.0, im_w_um )
ax.set_ylim( 0.0, im_h_um )
plt.xlabel("μm", fontsize=20)
plt.ylabel("μm", fontsize=20)

# plt.savefig('ztmp1.png', dpi=udpi)
plt.savefig(fname_z3, dpi=udpi)





#######################################################
plt.clf()

fig, ax = plt.subplots(figsize=ufigsize, dpi=udpi)
# ax.plot( xba, yba, 'o', markeredgecolor='#00ff00', markerfacecolor='#00000000' )
# ax.plot( xfa, yfa, 'o', markeredgecolor='#ff0000', markerfacecolor='#00000000' )
# ax.plot( gr2_xc, gr2_yc, '-', color='#000055')

for ci in range(n_cell):
  ax.plot( cellx_closed[ci], celly_closed[ci], '-', color='#777777' )


ax.plot( cell_cenfx, cell_cenfy, 'o', markeredgecolor='#000000', markerfacecolor='#ff9999' )
ax.plot( cell_cenpx, cell_cenpy, 'X', markeredgecolor='#000000', markerfacecolor='#000000ff' )


for ci in range(n_cell):
  ax.plot( gr_circ_x[ci], gr_circ_y[ci],
    '-', color='#00aa00',
    )
  ax.plot( gr_cell_mean_vx, gr_cell_mean_vy,
    '-', color='#0000aa',
    )



# plt.axis('equal')
ax.set_aspect('equal')
ax.set_xlim( 0.0, im_w_um )
ax.set_ylim( 0.0, im_h_um )
plt.xlabel("μm", fontsize=20)
plt.ylabel("μm", fontsize=20)

# plt.savefig('ztmp2.png', dpi=udpi)
plt.savefig(fname_z4, dpi=udpi)






