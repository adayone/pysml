import sys
for line in sys.stdin:
    line = line.strip()
    items = line.split(',')
    print '%s,%s,%s'%(items[0].zfill(6), items[1].zfill(6), 1-float(items[2]))
