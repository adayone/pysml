import  urllib2
def get_realtime(id):
    if id[0] == '6':
            cmd = "http://hq.sinajs.cn/list=sh%s"
    else:
        cmd = "http://hq.sinajs.cn/list=sz%s"
    if len(id) < 1:
        return 
    id = id.zfill(6)
    cmd = cmd%id
    rs = urllib2.urlopen(cmd)
    rs = rs.read()
    items = rs.split(',')
    if len(items) < 3:
        return ''
    name = items[0].strip().split('"')[1]
    now = float(items[3])
    begin = float(items[2])
    rat = round((now - begin)/begin, 4)
    if rat > 0:
        rat = "+{:.2%}".format(rat)
    else:
        rat = "{:.2%}".format(rat)
    return id, name, rat

