#!/usr/bin/python3

import math




#######################################################
def order_right_handed(polygonx, polygony, cenx, ceny):
  # Order points about cen.
  n_a = len(polygonx)
  #
  vcx = []
  vcy = []
  for i in range(n_a):
    vcx.append( polygonx[i]-cenx )
    vcy.append( polygony[i]-ceny )
  #
  ang = []
  for i in range(n_a):
    ang.append(0.0)
    ang[i] = math.atan2( vcy[i], vcx[i] )
    # atan2 return ang as:  -pi < ang < pi.
    # But we want:  0 < ang < 2pi.
    # Like that points start in quadrant I.
    if ang[i] < 0:  ang[i] += 2.0 * math.pi
  #
  zipped = list( zip(ang, polygonx, polygony) )
  zipped.sort()
  angs, sorx, sory = zip(*zipped)
  sorx = list(sorx)
  sory = list(sory)
  #
  # return vbs, cm
  return sorx, sory
#######################################################




#######################################################
# Check every point versus every triangle.
#
# Figure the cross product from every triangle edge
# to the vector from edge vector start point to the
# test point.
#
# If the cross products for every edge all have
# positive third components, the point is in the triangle.
#
# If a point is in any of the triangles, it's in the
# polygon.
#######################################################
def in_triangle(Tp1, Tp2, Tp3, test):
  edge1 = [Tp2[0]-Tp1[0], Tp2[1]-Tp1[1]]
  edge2 = [Tp3[0]-Tp2[0], Tp3[1]-Tp2[1]]
  edge3 = [Tp1[0]-Tp3[0], Tp1[1]-Tp3[1]]
  #
  s1 = [test[0]-Tp1[0], test[1]-Tp1[1]]
  s2 = [test[0]-Tp2[0], test[1]-Tp2[1]]
  s3 = [test[0]-Tp3[0], test[1]-Tp3[1]]
  #
  k1 = edge1[0] * s1[1] - edge1[1] * s1[0]  # edge1 x s1
  k2 = edge2[0] * s2[1] - edge2[1] * s2[0]  # edge2 x s2
  k3 = edge3[0] * s3[1] - edge3[1] * s3[0]  # edge3 x s3
  #
  if k1 >= 0 and k2 >= 0 and k3 >= 0:  return 1
  return 0
#######################################################




#######################################################
def in_polygon(polygonx, polygony, cenx, ceny, testx, testy):
  # Returns 1 if in poly, 0 if not.
  tri1 = []
  tri2 = []
  tri3 = []
  #
  n_a = len(polygonx)
  #
  for i in range(n_a):
    j = i+1
    if j == n_a:  j = 0
    #
    tri1.append( [cenx,ceny] )
    tri2.append( [polygonx[i], polygony[i]] )
    tri3.append( [polygonx[j], polygony[j]] )
  #
  inpoly = 0
  #
  for itri in range(n_a):
    vtest = [testx, testy]
    intri = in_triangle( tri1[itri], tri2[itri], tri3[itri], vtest )
    inpoly += intri
  #
  if inpoly > 0:  return 1
  return 0
#######################################################





#######################################################
def polygon_params(polygonx, polygony):
  # Need polygon = [[p0x,p0y], [p1x,p1y], ... [p(n-1)x, p(n-1)y]]
  n = len(polygonx)
  px = polygonx.copy()
  py = polygony.copy()
  px.append( px[0] )
  py.append( py[0] )
  #
  d = []
  for i in range(n):
    j = i + 1
    d.append( px[i] * py[j] - px[j] * py[i] )
  #
  area = 0.0
  for i in range(n):  area += d[i]
  area /= 2.0
  #
  cx = 0.0
  cy = 0.0
  for i in range(n):
    j = i + 1
    cx += (px[i]+px[j]) * d[i]
    cy += (py[i]+py[j]) * d[i]
  cx /= 6.0 * area
  cy /= 6.0 * area
  #
  return area, cx, cy
#######################################################






def get_centroid_of_points(x, y):
  cx = 0.0
  cy = 0.0
  n = len(x)
  for i in range(n):
    cx += x[i]
    cy += y[i]
  cx /= n
  cy /= n
  return cx, cy



def create_closed_contour(v):
  # Adds a copy of the first element of the list to the end.
  w = []
  n = len(v)
  for i in range(n):   w.append(v[i])
  w.append(v[0])
  return w


def create_closed_contour_arrays(p):
  # For each list in p...
  # Adds a copy of the first element of the list to the end.
  q = []
  n = len(p)
  for i in range(n):
    q.append([])
    #
    nn = len(p[i])
    #
    for j in range(nn):
      q[i].append( p[i][j] )
    #
    q[i].append( p[i][0] )
  #
  return q


def get_dv_unit(x1, y1, x2, y2):
  dux = x2 - x1
  duy = y2 - y1
  mag = math.sqrt( (dux**2) + (duy**2) )
  if mag > 0:
   dux /= mag
   duy /= mag
  #
  return dux, duy


def get_list_for_vec_graphing(p0x, p0y, p1x, p1y, scale):
  # Finds unit vectors p0x->p0y and scales them up to length scale.
  # Also adds NaNs between vector lines.
  qx = []
  qy = []
  n = len(p0x)
  for i in range(n):
    #
    mag = (p1x[i]-p0x[i])**2 + (p1y[i]-p0y[i])**2
    mag = math.sqrt(mag)
    dux = p1x[i] - p0x[i]
    duy = p1y[i] - p0y[i]
    if mag > 0.0:
      dux *= scale / mag
      duy *= scale / mag
    #
    qx.append( p0x[i] )
    qy.append( p0y[i] )
    #
    qx.append( p0x[i]+dux )
    qy.append( p0y[i]+duy )
    #
    qx.append( float('nan') )
    qy.append( float('nan') )
  return qx, qy


def get_list_for_vec_graphing_2(p0x, p0y, dx, dy, scale):
  # Creates arrays for graphing.
  # Each vector has endpoints:
  #   p0x   ->   pox + dx * scale
  #   p0y   ->   poy + dy * scale
  qx = []
  qy = []
  n = len(p0x)
  for i in range(n):
    #
    #
    qx.append( p0x[i] )
    qy.append( p0y[i] )
    #
    qx.append( p0x[i]+dx[i]*scale )
    qy.append( p0y[i]+dy[i]*scale )
    #
    qx.append( float('nan') )
    qy.append( float('nan') )
  return qx, qy




class vec2():
  def __init__(self, x=0.0, y=0.0):
    # self.x = 0.0
    # self.y = 0.0
    self.x = x
    self.y = y
  def set_xy(self,x,y):
    self.x = x
    self.y = y
  def mag2(self):
    return self.x**2 + self.y**2
  def mag(self):
    return math.sqrt( self.mag2() )
  def unit(self):
    u = vec2()
    m = self.mag()
    if m > 0.0:
      u.x = self.x/m
      u.y = self.y/m
    return u
  def multiply_scalar(self, a):
    self.x *= a
    self.y *= a
  def add_xy(self, dx, dy):
    self.x += dx
    self.y += dy
  def add_vec2(self, v):
    self.x += v.x
    self.y += v.y
  def get_ang(self):
    ang = math.atan2( self.y, self.x )
    return ang



class circle():
  def __init__(self):
    self.p = []
    self.n_pnt = 0
    self.n_seg = 0
    self.radius = 0.0
    self.p0 = vec2()
  def set(self, n_seg, p0, radius):
    self.n_seg = n_seg
    self.n_pnt = n_seg+1
    self.p0.x = p0.x
    self.p0.y = p0.y
    self.radius = radius
    dang = math.pi * 2.0 / self.n_seg
    for i in range(self.n_pnt):
      self.p.append( vec2() )
      self.p[i].x = self.p0.x + self.radius * math.cos(dang*i)
      self.p[i].y = self.p0.y + self.radius * math.sin(dang*i)
  def get_x_y_arrays(self):
    # Useful for plotting with matplotlib
    x = []
    y = []
    for i in range(self.n_pnt):
      x.append(self.p[i].x)
      y.append(self.p[i].y)
    return x, y




def dotproduct(a,b):
  return a.x * b.x + a.y + b.y



class bodyfoot():
  # Has a basal body and a basal foot.
  def __init__(self):
    self.body = vec2()  # position of body
    self.foot = vec2()  # position of foot
  def set_vec2(self, body, foot):
    self.body.x = body.x
    self.body.y = body.y
    self.foot.x = foot.x
    self.foot.y = foot.y
  def get_vec(self):
    v = vec2(self.foot.x - self.body.x, self.foot.y-self.body.y)
    return v
  def get_uvec(self):
    u = self.get_vec()
    m = u.mag()
    if m > 0.0:
      u.x /= m
      u.y /= m
    return u



def get_list_for_vec_graphing_3(b, scale):
  # Finds unit vectors and scales them up to length scale.
  # Also adds NaNs between vector lines.
  qx = []
  qy = []
  n = len(b)
  for i in range(n):
    v = b[i].get_uvec()
    v.multiply_scalar( scale )
    #
    qx.append( b[i].body.x )
    qy.append( b[i].body.y )
    qx.append( b[i].body.x + v.x )
    qy.append( b[i].body.y + v.y )
    #
    qx.append( float('nan') )
    qy.append( float('nan') )
  return qx, qy




def order_right_handed_2(p, cen):
  # Order points about cen.
  # p and cen need to be of class vec2
  n_a = len(p)
  #
  v = []
  for i in range(n_a):
    v.append( vec2(p[i].x-cen.x, p[i].y-cen.y) )
  #
  ang = []
  for i in range(n_a):
    ang.append(0.0)
    ang[i] = v[i].get_ang()
    # atan2 return ang as:  -pi < ang < pi.
    # But we want:  0 < ang < 2pi.
    # Like that points start in quadrant I.
    if ang[i] < 0:  ang[i] += 2.0 * math.pi
  #
  zipped = list( zip(ang, p) )
  zipped.sort()
  angs, sp = zip(*zipped)
  sp = list(sp)
  #
  return sp




# def in_polygon2(polygonx, polygony, cenx, ceny, testx, testy):
def in_polygon2(pg, cen, p):
  # Returns 1 if in poly, 0 if not.
  # pg:  polygon vec2 array
  # cen: center vec2
  # p:  test point vec2
  tri1 = []
  tri2 = []
  tri3 = []
  #
  n_a = len(pg)
  #
  for i in range(n_a):
    j = i+1
    if j == n_a:  j = 0
    #
    tri1.append( [cen.x,cen.y] )
    tri2.append( [pg[i].x, pg[i].y] )
    tri3.append( [pg[j].x, pg[j].y] )
  #
  inpoly = 0
  #
  for itri in range(n_a):
    vtest = [p.x, p.y]
    intri = in_triangle( tri1[itri], tri2[itri], tri3[itri], vtest )
    inpoly += intri
  #
  if inpoly > 0:  return 1
  return 0
#######################################################


#######################################################
def polygon_params_2(p):
  # p = polygon points as vec2 array
  # Need polygon = [[p0x,p0y], [p1x,p1y], ... [p(n-1)x, p(n-1)y]]
  n = len(p)
  #
  pp = []
  for i in range(n):
    pp.append( vec2() )
    pp[i].set_xy( p[i].x, p[i].y )
  pp.append( vec2() )
  pp[n].set_xy( p[0].x, p[0].y )
  #
  d = []
  for i in range(n):
    j = i + 1
    d.append( pp[i].x * pp[j].y - pp[j].x * pp[i].y )
  #
  area = 0.0
  for i in range(n):  area += d[i]
  area /= 2.0
  #
  cen = vec2()
  for i in range(n):
    j = i + 1
    #
    cen.x += (pp[i].x+pp[j].x) * d[i]
    cen.y += (pp[i].y+pp[j].y) * d[i]
  cen.x /= 6.0 * area
  cen.y /= 6.0 * area
  #
  # Return float area and vec2 center of plane figure
  return area, cen
#######################################################






class pogocell():
  # polygon cell, a cell defined by a polygon.
  def __init__(self):
    self.p = []    # polygon points
    self.n_p = 0
    self.pp = []   # polygon points + closing point
    self.n_pp = 0  # usually n_p + 1
    self.pcent = vec2()   # center of points
    self.fcent = vec2()   # center of plane figure
    self.area = 0.0
  def add_point_from_str(self, s1, s2):
    p1 = float(s1)
    p2 = float(s2)
    self.p.append( vec2(p1, p2) )
    self.n_p += 1
  def rescale(self, scale):
    for i in range(self.n_p):
      self.p[i].x *= scale
      self.p[i].y *= scale
  def apply_invy(self, h):
    for i in range(self.n_p):
      self.p[i].y = h - self.p[i].y
  def calc_pcent(self):
    self.pcent.set_xy(0.0,0.0)
    for i in range(self.n_p):
      self.pcent.x += self.p[i].x
      self.pcent.y += self.p[i].y
    self.pcent.multiply_scalar( 1.0 / float(self.n_p) )
  def calc_fcent(self):
    self.area, self.fcent = polygon_params_2(self.p)
  def sort_points(self):
    self.calc_pcent()
    self.p = order_right_handed_2(self.p, self.pcent)
  def get_closed_countour_arrays(self):
    x = []
    y = []
    for i in range(self.n_p):
      x.append( self.p[i].x )
      y.append( self.p[i].y )
    x.append( self.p[0].x )
    y.append( self.p[0].y )
    return x, y
    ####
  def in_cell(self, test):
    # test is the point to test
    # it is a vec2
    r = in_polygon2(self.p, self.pcent, test)
    return r



