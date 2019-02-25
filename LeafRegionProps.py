class LeafRegionProps:
  PETIOLE = 0
  BLADE = 1
  def __init__(self):    
    self.label = None
    self.centroid = None 
    self.centroidZ = None
    self.area2D = 0
    self.area3D = 0
    self.meanBladeInclination = 0 
    self.medianBladeInclination = 0 
    self.pointBasedBladeInclination = 0 
    self.bladeWidth2D=0
    self.bladeLength2D=0
    self.bladeLength3D=0 
    self.bladeLengthPointBased2D=0
    self.bladeLengthPointBased3D=0
    self.distFromCentre =0
    self.meanPetioleInclination = 0
    self.medianPetioleInclination = 0
    self.pointBasedPetioleInclination = 0 
    self.petioleLength2D=0
    self.petioleLength2DRP=0
    self.petioleWidth2D=0
    self.petioleLength3D=0
    
    self.bladeRegionprops=None 
    self.bladeAndPetioleRegionprops=None 
    self.bladeInsertionCoord = None 
    self.bladeTipCoord = None 
    self.petioleStartCoord = None 
    
    self.bladeMask = None
    self.petioleMask = None
    self.leafMask = None
    self.sn= None