# insert_code.gdb
# Mô tả: Script GDB để chèn mã vào tiến trình tại điểm breakpoint.
# Ghi chú: Đảm bảo rằng hàm custom_code() đã được định nghĩa trong tiến trình mục tiêu.

# Ghi log bắt đầu chèn mã
printf "=== Start Code Injection ===\n" >> /app/logs/inject_code.log

# Chèn mã bằng cách gọi hàm custom_code()
call custom_code()
if $_last_exit_code != 0
    printf "Error: Gọi hàm custom_code() thất bại\n" >> /app/logs/inject_code.log
    quit
end

# Ghi log sau khi chèn mã thành công
printf "Chèn mã thành công tại breakpoint\n" >> /app/logs/inject_code.log

# Tiếp tục tiến trình
continue
