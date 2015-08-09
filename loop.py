import sys
import os
f = open('pred_min', 'w')
for id in sys.stdin:
    id = id.strip()
    print id
    rs = os.popen('python model_main.py %s'%id).read()
    print rs
    f.write(rs)
    f.flush()
f.close()



