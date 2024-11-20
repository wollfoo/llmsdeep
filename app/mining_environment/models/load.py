import tensorflow as tf

try:
    model = tf.keras.models.load_model('cloaking_model.keras')
    print("Mô hình đã được tải thành công.")
    model.summary()
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
