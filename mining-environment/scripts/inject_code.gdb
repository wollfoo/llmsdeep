# inject_code.gdb
# Mô tả: Script GDB để thực hiện code injection vào tiến trình mục tiêu.
# Ghi chú: Đảm bảo rằng bạn có quyền cần thiết để đính kèm vào tiến trình.

# Thiết lập biến cấu hình
set $pid = TARGET_PID                    # Thay thế TARGET_PID bằng PID thực tế
set $injection_script = "injection_script/insert_code.gdb"  # Thay thế bằng đường dẫn thực tế tới script injection
set $breakpoint_function = "BREAKPOINT_FUNCTION"  # Thay thế bằng tên hàm hoặc địa chỉ breakpoint

# Ghi log bắt đầu quá trình injection
printf "=== Injection Script Started ===\n" >> /app/logs/inject_code.log

# Đính kèm vào tiến trình mục tiêu
attach $pid
if $_last_exit_code != 0
    printf "Error: Không thể đính kèm tới PID %d\n" $pid >> /app/logs/inject_code.log
    detach
    quit
end

# Ghi log sau khi đính kèm thành công
printf "Đã đính kèm tới tiến trình PID %d\n" $pid >> /app/logs/inject_code.log

# Đặt breakpoint tại hàm mục tiêu
break $breakpoint_function
if $_last_exit_code != 0
    printf "Error: Không thể đặt breakpoint tại %s\n" $breakpoint_function >> /app/logs/inject_code.log
    detach
    quit
end

# Ghi log sau khi đặt breakpoint thành công
printf "Đã đặt breakpoint tại %s\n" $breakpoint_function >> /app/logs/inject_code.log

# Tiếp tục tiến trình để đạt breakpoint
continue
if $_last_exit_code != 0
    printf "Error: Tiếp tục tiến trình không thành công\n" >> /app/logs/inject_code.log
    detach
    quit
end

# Ghi log khi tiến trình dừng tại breakpoint
printf "Tiến trình dừng tại breakpoint %s\n" $breakpoint_function >> /app/logs/inject_code.log

# Chèn mã vào bộ nhớ tiến trình bằng cách chạy script injection
source $injection_script
if $_last_exit_code != 0
    printf "Error: Chèn mã không thành công từ script %s\n" $injection_script >> /app/logs/inject_code.log
    detach
    quit
end

# Ghi log sau khi chèn mã thành công
printf "Chèn mã thành công từ script %s\n" $injection_script >> /app/logs/inject_code.log

# Tiếp tục tiến trình sau khi injection
continue
if $_last_exit_code != 0
    printf "Error: Tiếp tục tiến trình sau injection không thành công\n" >> /app/logs/inject_code.log
    detach
    quit
end

# Ghi log kết thúc quá trình injection
printf "=== Injection Script Completed Successfully ===\n" >> /app/logs/inject_code.log

# Tách khỏi tiến trình
detach

# Thoát GDB
quit
