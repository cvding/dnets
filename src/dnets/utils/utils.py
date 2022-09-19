from typing import OrderedDict


def load_wgts(net, state_dict, strict=True, show=True):
    """加载权重

    Args:
        net (nn.Module): 输入网络
        state_dict (dict): 从torch.load()加载的权重
        strict (bool, optional): 是否严格匹配字典. Defaults to True.
        show (bool, optional): 是否显示信息. Defaults to True.

    Raises:
        ValueError: 如果严格控制，不能加载部分权重

    Returns:
        nn.Module: 输出网络
    """
    device = next(net.parameters()).device
    dst = net.state_dict()
    src = state_dict if 'state_dict' not in state_dict else state_dict['state_dict']

    new_src = OrderedDict()
    for k, v in src.items():
        if 'module.' == k[0:7]:
            name = k[7:]
        else:
            name = k
        new_src[name] = v
    src = new_src

    pretrained_dict = {}
    for k, v in dst.items():
        mk = k[7:]
        if k in src and src[k].size() == v.size():
            # k in state_dict (same as net)
            pretrained_dict[k] = src[k].to(device)
        elif mk in src and src[mk].size() == v.size():
            # mk in state_dict (not same as net)
            pretrained_dict[k] = src[mk].to(device)
        else:
            pass

    if len(pretrained_dict) == len(dst):
        print("%s : All parameters loading." % type(net).__name__)
    else:
        nkey = 0
        not_loaded_keys = []
        for k in dst.keys():
            if k not in pretrained_dict.keys():
                not_loaded_keys.append(k)
                print("dst:", k, dst[k].size())
            else:
                nkey += 1
        for k in src.keys():
            if k not in pretrained_dict.keys():
                print("src:", k, src[k].size())
        
        if show:
            print('%s: Some params were not loaded.' % type(net).__name__)
            print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
            print("load %d keys of %d" % (len(pretrained_dict), len(src)))

        if strict:
            raise ValueError("Strict Parameter load failed")

        if nkey == 0:
            return None 

    dst.update(pretrained_dict)
    net.load_state_dict(dst)

    return net