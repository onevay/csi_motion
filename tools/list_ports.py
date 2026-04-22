import tools.list_ports

ports = tools.list_ports.comports()
for p in ports:
    print(p.device, p.description)
