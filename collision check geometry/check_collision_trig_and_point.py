import numpy as np
# http://www.jeffreythompson.org/collision-detection/tri-point.php

x1 = 0
y1 = 0

x2 = 5
y2 = 0

x3 = 2.5
y3 = 2.5

px = 0
py = 0


areaOrig = abs( (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) )
area1 =    abs( (x1-px)*(y2-py) - (x2-px)*(y1-py) )
area2 =    abs( (x2-px)*(y3-py) - (x3-px)*(y2-py) )
area3 =    abs( (x3-px)*(y1-py) - (x1-px)*(y3-py) )

if (area1+area2+area3== areaOrig):
    print("yes")
else:
    print("No")