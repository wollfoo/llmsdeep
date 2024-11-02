import json
import logging
import subprocess
import sys
import os
from pathlib import Path
import psutil
from elftools.elf.elffile import ELFFile
import requests

# Thiết lập logging
logging.basicConfig(
    filename='/app/logs/inject_code.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def log(message):
    """Ghi log vào tệp."""
    logging.info(message)
    print(message)  # Optional: Hiển thị log trên console

def load_config(config_path):
    """Tải tệp cấu hình."""
    config_file = Path(config_path)
    if not config_file.is_file():
        log(f"Lỗi: Tệp cấu hình không tồn tại: {config_path}")
        sys.exit(1)
    
    try:
        with config_file.open('r') as file:
            config = json.load(file)
        log(f"Đã tải cấu hình từ {config_path}")
        return config
    except json.JSONDecodeError as e:
        log(f"Lỗi khi phân tích tệp cấu hình: {e}")
        sys.exit(1)

def find_pid(target_process_name):
    """Tìm PID của tiến trình mục tiêu."""
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == target_process_name:
            log(f"Tìm thấy tiến trình mục tiêu: {target_process_name} với PID: {proc.info['pid']}")
            return proc.info['pid']
    log(f"Không tìm thấy tiến trình mục tiêu: {target_process_name}")
    return None

def get_binary_path(pid):
    """Xác định đường dẫn tệp nhị phân của tiến trình."""
    try:
        proc = psutil.Process(pid)
        binary_path = proc.exe()
        log(f"Đường dẫn tệp nhị phân của PID {pid}: {binary_path}")
        return binary_path
    except Exception as e:
        log(f"Lỗi khi xác định đường dẫn tệp nhị phân cho PID {pid}: {e}")
        return None

def list_functions(binary_path):
    """Liệt kê các hàm trong tệp nhị phân sử dụng pyelftools."""
    functions = []
    try:
        with open(binary_path, 'rb') as f:
            elffile = ELFFile(f)
            if not elffile.has_dwarf_info():
                log("Tệp nhị phân không có thông tin DWARF. Các hàm có thể không đầy đủ.")
            symtab = elffile.get_section_by_name('.symtab')
            if not symtab:
                log("Không tìm thấy bảng ký hiệu trong tệp nhị phân.")
                return functions
            for symbol in symtab.iter_symbols():
                if symbol['st_info']['type'] == 'STT_FUNC':
                    functions.append(symbol.name)
        log(f"Liệt kê {len(functions)} hàm trong tệp nhị phân.")
        return functions
    except Exception as e:
        log(f"Lỗi khi liệt kê các hàm trong tệp nhị phân: {e}")
        return functions

def select_breakpoint_function(functions):
    """Chọn hàm mục tiêu để đặt breakpoint."""
    # Tiêu chí chọn hàm: Ví dụ, chọn hàm 'main' nếu có
    if 'main' in functions:
        log("Chọn hàm 'main' làm breakpoint function.")
        return 'main'
    # Nếu không có 'main', có thể chọn hàm đầu tiên hoặc theo tiêu chí khác
    elif functions:
        log(f"Chọn hàm '{functions[0]}' làm breakpoint function.")
        return functions[0]
    else:
        log("Không tìm thấy hàm nào để đặt breakpoint.")
        return None

def generate_gdb_script(pid, injection_script_path, gdb_script_path='/app/scripts/injection_scripts/inject_code.gdb'):
    """Tạo script GDB với thông tin đã xác định."""
    try:
        with open(gdb_script_path, 'w') as gdb_file:
            gdb_file.write("# inject_code.gdb\n")
            gdb_file.write("# Mô tả: Script GDB để thực hiện code injection vào tiến trình mục tiêu.\n")
            gdb_file.write("# Ghi chú: Đảm bảo rằng bạn có quyền cần thiết để đính kèm vào tiến trình.\n\n")
            
            # Thiết lập biến cấu hình
            gdb_file.write(f'set $pid = {pid}\n')
            gdb_file.write(f'set $injection_script = "{injection_script_path}"\n')
            gdb_file.write(f'set $breakpoint_function = "main"\n\n')  # Chọn 'main' hoặc hàm khác đã chọn
            
            # Ghi log bắt đầu quá trình injection
            gdb_file.write('printf "=== Injection Script Started ===\\n" >> /app/logs/inject_code.log\n\n')
            
            # Đính kèm vào tiến trình mục tiêu
            gdb_file.write('attach $pid\n')
            gdb_file.write('if $_last_exit_code != 0\n')
            gdb_file.write('    printf "Error: Không thể đính kèm tới PID %d\\n" $pid >> /app/logs/inject_code.log\n')
            gdb_file.write('    detach\n')
            gdb_file.write('    quit\n')
            gdb_file.write('end\n\n')
            
            # Ghi log sau khi đính kèm thành công
            gdb_file.write('printf "Đã đính kèm tới tiến trình PID %d\\n" $pid >> /app/logs/inject_code.log\n\n')
            
            # Đặt breakpoint tại hàm mục tiêu
            gdb_file.write('break $breakpoint_function\n')
            gdb_file.write('if $_last_exit_code != 0\n')
            gdb_file.write('    printf "Error: Không thể đặt breakpoint tại %s\\n" $breakpoint_function >> /app/logs/inject_code.log\n')
            gdb_file.write('    detach\n')
            gdb_file.write('    quit\n')
            gdb_file.write('end\n\n')
            
            # Ghi log sau khi đặt breakpoint thành công
            gdb_file.write('printf "Đã đặt breakpoint tại %s\\n" $breakpoint_function >> /app/logs/inject_code.log\n\n')
            
            # Tiếp tục tiến trình để đạt breakpoint
            gdb_file.write('continue\n')
            gdb_file.write('if $_last_exit_code != 0\n')
            gdb_file.write('    printf "Error: Tiếp tục tiến trình không thành công\\n" >> /app/logs/inject_code.log\n')
            gdb_file.write('    detach\n')
            gdb_file.write('    quit\n')
            gdb_file.write('end\n\n')
            
            # Ghi log khi tiến trình dừng tại breakpoint
            gdb_file.write('printf "Tiến trình dừng tại breakpoint %s\\n" $breakpoint_function >> /app/logs/inject_code.log\n\n')
            
            # Tải shared library chứa custom_code() bằng cách gọi dlopen
            gdb_file.write('call dlopen("/app/scripts/injection_scripts/libcustomcode.so", 1)\n')  # RTLD_NOW = 1
            gdb_file.write('if $_last_exit_code == 0\n')
            gdb_file.write('    printf "Đã tải shared library thành công.\n" >> /app/logs/inject_code.log\n')
            gdb_file.write('else\n')
            gdb_file.write('    printf "Error: Không thể tải shared library.\n" >> /app/logs/inject_code.log\n')
            gdb_file.write('    detach\n')
            gdb_file.write('    quit\n')
            gdb_file.write('end\n\n')
            
            # Gọi hàm custom_code() từ shared library
            gdb_file.write('call custom_code()\n')
            gdb_file.write('if $_last_exit_code != 0\n')
            gdb_file.write('    printf "Error: Gọi hàm custom_code() thất bại\\n" >> /app/logs/inject_code.log\n')
            gdb_file.write('    detach\n')
            gdb_file.write('    quit\n')
            gdb_file.write('end\n\n')
            
            # Ghi log sau khi chèn mã thành công
            gdb_file.write('printf "Chèn mã thành công tại breakpoint\n" >> /app/logs/inject_code.log\n\n')
            
            # Tiếp tục tiến trình sau khi injection
            gdb_file.write('continue\n')
            gdb_file.write('if $_last_exit_code != 0\n')
            gdb_file.write('    printf "Error: Tiếp tục tiến trình sau injection không thành công\\n" >> /app/logs/inject_code.log\n')
            gdb_file.write('    detach\n')
            gdb_file.write('    quit\n')
            gdb_file.write('end\n\n')
            
            # Ghi log kết thúc quá trình injection
            gdb_file.write('printf "=== Injection Script Completed Successfully ===\\n" >> /app/logs/inject_code.log\n\n')
            
            # Tách khỏi tiến trình
            gdb_file.write('detach\n\n')
            
            # Thoát GDB
            gdb_file.write('quit\n')
        
        log(f"Tạo script GDB tại {gdb_script_path}")
        return True
    except Exception as e:
        log(f"Lỗi khi tạo script GDB: {e}")
        return False

def inject_code_into_process(pid, gdb_script_path):
    """Tiêm mã vào tiến trình mục tiêu bằng GDB."""
    if not Path(gdb_script_path).is_file():
        log(f"Lỗi: Không tìm thấy tệp script GDB tại {gdb_script_path}")
        return False
    
    try:
        cmd = ['sudo', 'gdb', '-batch', '-x', gdb_script_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log(f"Tiêm mã thành công vào tiến trình PID {pid}")
            return True
        log(f"Lỗi tiêm mã vào tiến trình PID {pid}: {result.stderr}")
        return False
    except Exception as e:
        log(f"Lỗi khi tiêm mã vào tiến trình PID {pid}: {e}")
        return False

def notify_cloaking_module(cloaking_endpoint, pid):
    """Thông báo cho module cloaking sau khi injection thành công."""
    try:
        payload = {'pid': pid}
        response = requests.post(cloaking_endpoint, json=payload, timeout=5)
        if response.status_code == 200:
            log(f"Đã thông báo cloaking thành công cho PID: {pid}")
        else:
            log(f"Lỗi khi thông báo cloaking cho PID: {pid}: {response.text}")
    except requests.RequestException as e:
        log(f"Lỗi khi gửi thông tin đến cloaking module: {e}")

def main():
    config_path = '/app/config/injection_config.json'
    config = load_config(config_path)
    
    target_process_name = config.get('target_process')
    injection_script = config.get('injection_script')
    cloaking_endpoint = config.get('cloaking_endpoint')
    
    if not all([target_process_name, injection_script, cloaking_endpoint]):
        log("Lỗi: Các tham số cấu hình không đầy đủ.")
        sys.exit(1)
    
    pid = find_pid(target_process_name)
    if not pid:
        log("Không thể tìm thấy tiến trình mục tiêu, kết thúc script.")
        sys.exit(1)
    
    binary_path = get_binary_path(pid)
    if not binary_path:
        log("Không thể xác định đường dẫn tệp nhị phân của tiến trình, kết thúc script.")
        sys.exit(1)
    
    functions = list_functions(binary_path)
    if not functions:
        log("Không tìm thấy hàm nào trong tệp nhị phân, kết thúc script.")
        sys.exit(1)
    
    breakpoint_function = select_breakpoint_function(functions)
    if not breakpoint_function:
        log("Không thể xác định hàm mục tiêu để đặt breakpoint, kết thúc script.")
        sys.exit(1)
    
    # Đường dẫn đến script GDB chính
    gdb_script_path = '/app/scripts/injection_scripts/inject_code.gdb'
    
    # Đường dẫn đến shared library chứa custom_code()
    shared_library_path = '/app/scripts/injection_scripts/libcustomcode.so'
    
    # Tạo script GDB với thông tin đã xác định
    if not generate_gdb_script(pid, shared_library_path, gdb_script_path):
        log("Tạo script GDB thất bại, kết thúc script.")
        sys.exit(1)
    
    # Tiêm mã vào tiến trình bằng script GDB đã tạo
    if inject_code_into_process(pid, gdb_script_path):
        notify_cloaking_module(cloaking_endpoint, pid)
    else:
        log("Tiêm mã thất bại, không thông báo cloaking.")
        sys.exit(1)

if __name__ == "__main__":
    main()
