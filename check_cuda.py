import torch
print(f'{torch.cuda.is_available()=}')
print(f'{torch.cuda.get_device_name(0)=}')
print('Memory Usage:')
free, total = torch.cuda.mem_get_info(device=0)
total = total / 1024 ** 3
free = free / 1024 ** 3
used = total - free
print('USED -- FREE -- TOTAL')
print('=====================')
print(f'{round(used,2)}, {round(free,2)},  {round(total,2)} in GB')
