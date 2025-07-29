import platform


def get_windows_cpu_name():
    import subprocess
    # Try WMIC
    try:
        output = subprocess.check_output(['wmic', 'cpu', 'get', 'Name'], encoding='utf-8')
        lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
        if len(lines) > 1:
            cpu_name = lines[1]
            if 'Family' not in cpu_name and 'GenuineIntel' not in cpu_name:
                return cpu_name
    except Exception:
        pass
    # Try PowerShell
    try:
        output = subprocess.check_output([
            'powershell', '-Command',
            'Get-CimInstance -ClassName Win32_Processor | Select-Object -ExpandProperty Name'
        ], encoding='utf-8')
        cpu_name = output.strip().split('\n')[0].strip()
        if cpu_name:
            return cpu_name
    except Exception:
        pass
    return None


def get_cpu_name():
    import platform
    try:
        if platform.system() == 'Windows':
            cpu_name = get_windows_cpu_name()
            if cpu_name:
                return cpu_name
        import cpuinfo
        cpu_name = cpuinfo.get_cpu_info().get('brand_raw')
        if not cpu_name:
            cpu_name = cpuinfo.get_cpu_info().get('brand')
        if cpu_name:
            return cpu_name
    except Exception:
        pass
    return platform.processor() or platform.uname().processor or "Generic CPU"


def get_available_devices(backend='pytorch'):
    devices = []
    if backend == 'pytorch':
        import torch
        cpu_name = get_cpu_name()
        devices.append({'name': cpu_name, 'type': 'CPU', 'id': None})
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append({'name': torch.cuda.get_device_name(i), 'type': 'GPU', 'id': i})
    elif backend == 'openvino':
        try:
            from openvino import Core
            core = Core()
            for dev in core.available_devices:
                dev_type = get_device_type_openvino(dev)
                devices.append({'name': dev, 'type': dev_type, 'id': None})
        except ImportError:
            print("OpenVINO is not installed.")
    elif backend == 'onnx':
        devices.append({'name': 'CPU', 'type': 'CPU', 'id': None})
    return devices


def get_device_type_openvino(dev_name):
    if 'CPU' in dev_name:
        return 'CPU'
    if 'GPU' in dev_name:
        return 'GPU'
    if 'NPU' in dev_name:
        return 'NPU'
    if 'VPU' in dev_name:
        return 'VPU'
    return 'OTHER'


def print_available_devices(backend='pytorch'):
    devices = get_available_devices(backend)
    print(f"Available devices for backend '{backend}':")
    for d in devices:
        id_str = f" (id {d['id']})" if d['id'] is not None else ""
        print(f"- {d['name']} [{d['type']}{id_str}]")


def select_main_device(devices):
    # Priority: GPU > NPU > VPU > CPU > other
    for dev_type in ['GPU', 'NPU', 'VPU', 'CPU']:
        for d in devices:
            if d['type'] == dev_type:
                return get_device_object(d)
    # If none of the above, return the first one
    d = devices[0]
    return get_device_object(d)


def get_device_object(d):
    # Returns the appropriate device object according to the type
    if d['type'] == 'GPU':
        try:
            import torch
            return torch.device(f"cuda:{d['id']}") if d['id'] is not None else torch.device('cuda'), d['name'], d[
                'type'], d['id']
        except ImportError:
            pass
    if d['type'] == 'CPU':
        try:
            import torch
            return torch.device('cpu'), d['name'], d['type'], d['id']
        except ImportError:
            pass
    # For NPU, VPU or others, just return the name
    return d['name'], d['name'], d['type'], d['id']


def get_eval_devices(devices):
    eval_devices = []
    for d in devices:
        if d['type'] == 'GPU':
            try:
                import torch
                eval_devices.append(torch.device(f"cuda:{d['id']}") if d['id'] is not None else torch.device('cuda'))
            except ImportError:
                eval_devices.append(d['name'])
        elif d['type'] == 'CPU':
            try:
                import torch
                eval_devices.append(torch.device('cpu'))
            except ImportError:
                eval_devices.append(d['name'])
        elif d['type'] in ['NPU', 'VPU']:
            eval_devices.append(d['name'])
    return list({str(dev): dev for dev in eval_devices}.values())


def _get_fullname_from_torch_device(dev, devices):
    if dev.type == 'cuda':
        dev_index = getattr(dev, 'index', None)
        for d in devices:
            if isinstance(d, dict) and d.get('type') == 'GPU':
                if dev_index is not None and d.get('id') == dev_index:
                    return d.get('name')
                if dev_index is None:
                    return d.get('name')
    elif dev.type == 'cpu':
        for d in devices:
            if isinstance(d, dict) and d.get('type') == 'CPU':
                return get_cpu_name()
    return None


def _get_fullname_from_str_device(dev, devices):
    for d in devices:
        if isinstance(d, dict) and d.get('name') == dev:
            return d.get('name')
    return None


def _get_fallback_fullname(dev):
    if hasattr(dev, 'type'):
        if dev.type == 'cuda':
            try:
                import torch
                return torch.cuda.get_device_name(dev)
            except Exception:
                pass
        elif dev.type == 'cpu':
            return get_cpu_name()
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get('brand_raw')
    except Exception:
        return platform.processor() or platform.uname().processor or 'CPU'


def get_device_fullname(dev, devices=None):
    """
    Returns the real commercial name of the device (CPU, GPU, NPU, etc) using the detected devices list.
    Uses get_cpu_name logic for CPUs in Windows.
    """
    if devices is not None:
        if hasattr(dev, 'type'):
            name = _get_fullname_from_torch_device(dev, devices)
            if name:
                return name
        elif isinstance(dev, str):
            name = _get_fullname_from_str_device(dev, devices)
            if name:
                return name
    return _get_fallback_fullname(dev)
