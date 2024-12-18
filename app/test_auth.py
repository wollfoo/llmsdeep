import os
import logging
from azure.identity import ClientSecretCredential

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_authentication():
    client_id = os.getenv('AZURE_CLIENT_ID')
    client_secret = os.getenv('AZURE_CLIENT_SECRET')
    tenant_id = os.getenv('AZURE_TENANT_ID')

    # Loại bỏ khoảng trắng và ký tự không mong muốn
    tenant_id = tenant_id.strip() if tenant_id else None
    client_id = client_id.strip() if client_id else None
    client_secret = client_secret.strip() if client_secret else None

    # Kiểm tra giá trị biến môi trường với repr để thấy các ký tự đặc biệt
    logger.debug(f"AZURE_SUBSCRIPTION_ID: {repr(os.getenv('AZURE_SUBSCRIPTION_ID'))}")
    logger.debug(f"AZURE_CLIENT_ID: {repr(client_id)}")
    logger.debug(f"AZURE_TENANT_ID: {repr(tenant_id)}")

    if not all([client_id, client_secret, tenant_id]):
        logger.error("AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, và AZURE_TENANT_ID phải được thiết lập.")
        return

    # Kiểm tra định dạng GUID của Tenant ID
    if not is_valid_guid(tenant_id):
        logger.error("AZURE_TENANT_ID không hợp lệ định dạng GUID.")
        return

    try:
        credential = ClientSecretCredential(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id
        )
        logger.info("Đã xác thực thành công với Azure AD.")
    except Exception as e:
        logger.error(f"Lỗi khi xác thực với Azure AD: {e}")

def is_valid_guid(guid: str) -> bool:
    import re
    regex = re.compile(
        r'^[a-fA-F0-9]{8}\-'
        r'[a-fA-F0-9]{4}\-'
        r'[a-fA-F0-9]{4}\-'
        r'[a-fA-F0-9]{4}\-'
        r'[a-fA-F0-9]{12}$'
    )
    return bool(regex.match(guid))

if __name__ == "__main__":
    test_authentication()
