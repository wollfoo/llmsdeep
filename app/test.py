import logging

class AzureTrafficAnalyticsClient(AzureBaseClient):
    """
    Lớp hỗ trợ thao tác với Traffic Analytics trên Azure,
    bao gồm bật Traffic Analytics (nếu chưa bật) và truy vấn dữ liệu.
    """

    def __init__(self, logger: logging.Logger):
        """
        Khởi tạo AzureTrafficAnalyticsClient với logger đã cung cấp.
        Dùng AzureBaseClient để có credential và subscription_id.
        """
        super().__init__(logger)
        self.credential = DefaultAzureCredential()   # Thay thế bằng InteractiveBrowserCredential
        self.log_analytics_client = LogsQueryClient(self.credential)
        self.network_client = NetworkManagementClient(self.credential, self.subscription_id)
        # Lấy danh sách tất cả Log Analytics Workspaces (không cần biết tên)
        self.workspace_ids: List[str] = self.get_traffic_workspace_ids()

    def get_traffic_workspace_ids(self) -> List[str]:
        """
        Khám phá và trả về danh sách resource IDs của TẤT CẢ Log Analytics Workspaces
        trong subscription, không cần biết tên trước.
        """
        try:
            # 1) Dùng discover_resources để lấy tất cả Workspace (type = 'Microsoft.OperationalInsights/workspaces')
            resources = self.discover_resources('Microsoft.OperationalInsights/workspaces')
            
            # 2) Lọc lấy ID của từng workspace (nếu có)
            workspace_ids = [
                res['id']
                for res in resources
                if 'id' in res
            ]

            # 3) Ghi log về số lượng tìm được
            if workspace_ids:
                self.logger.info(f"Đã tìm thấy {len(workspace_ids)} Log Analytics Workspaces.")
            else:
                self.logger.warning("Không tìm thấy Log Analytics Workspace nào.")

            return workspace_ids

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy Workspace IDs của Log Analytics: {e}")
            return []

    def get_valid_timespan(self, workspace_id: str) -> Optional[tuple]:
        query = """
        AzureNetworkAnalytics_CL
        | summarize MinTime=min(TimeGenerated), MaxTime=max(TimeGenerated)
        """
        try:
            response = self.log_analytics_client.query_workspace(
                workspace_id=workspace_id,
                query=query,
                timespan=None  # Không giới hạn timespan
            )

            if response.tables and response.tables[0].rows:
                min_time, max_time = response.tables[0].rows[0]
                min_time = min_time.replace(tzinfo=timezone.utc)
                max_time = max_time.replace(tzinfo=timezone.utc)
                self.logger.info(f"Khoảng thời gian hợp lệ: {min_time} - {max_time}")
                return min_time, max_time
            else:
                self.logger.warning(f"Bảng AzureNetworkAnalytics_CL không có dữ liệu.")
                return None

        except HttpResponseError as e:
            if "PathNotFoundError" in e.message:
                self.logger.error(f"Bảng AzureNetworkAnalytics_CL không tồn tại: {e.message}")
            else:
                self.logger.error(f"Lỗi HTTP không xác định: {e.message}")
            if e.response:
                self.logger.error(f"Chi tiết phản hồi API: {e.response.json()}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy khoảng thời gian hợp lệ: {e}")
            return None
    
    def is_table_available(self, workspace_id: str) -> bool:
        query = """
        AzureNetworkAnalytics_CL
        | take 1
        """
        try:
            response = self.log_analytics_client.query_workspace(
                workspace_id=workspace_id,
                query=query,
                timespan=None
            )
            return bool(response.tables and response.tables[0].rows)
        except HttpResponseError as e:
            self.logger.error(f"Lỗi khi kiểm tra bảng AzureNetworkAnalytics_CL: {e.message}")
            return False


    def get_traffic_data(self, query: Optional[str] = None, timespan: Optional[Any] = None) -> List[LogsQueryResult]:
        results: List[LogsQueryResult] = []

        if not self.workspace_ids:
            self.logger.error("Không có Workspace ID để lấy dữ liệu.")
            return results

        try:
            if not query:
                query = """
                AzureNetworkAnalytics_CL
                | summarize ConnectionCount = count() by DestinationIP_s, DestinationPort, bin(TimeGenerated, 1m)
                | where ConnectionCount > 100
                """

            for workspace_id in self.workspace_ids:
                if not self.is_table_available(workspace_id):
                    self.logger.warning(f"Bảng AzureNetworkAnalytics_CL không tồn tại trong Workspace ID: {workspace_id}")
                    continue

                if not timespan:
                    self.logger.info(f"Tự động xác định timespan cho Workspace ID: {workspace_id}")
                    valid_timespan = self.get_valid_timespan(workspace_id)
                    if not valid_timespan:
                        self.logger.warning(f"Không thể xác định timespan cho Workspace ID: {workspace_id}, sử dụng mặc định.")
                        end_time = datetime.utcnow().replace(tzinfo=timezone.utc)
                        start_time = end_time - timedelta(hours=1)
                        timespan = (start_time, end_time)

                self.logger.info(f"Truy vấn với timespan: {timespan}")
                self.logger.info(f"Query: {query}")

                try:
                    response = self.log_analytics_client.query_workspace(
                        workspace_id=workspace_id,
                        query=query,
                        timespan=timespan
                    )

                    if response.tables:
                        results.extend(response.tables)
                        self.logger.info(f"Đã lấy dữ liệu từ Workspace ID: {workspace_id}")
                    else:
                        self.logger.info(f"Không có dữ liệu trả về từ Workspace ID: {workspace_id}")
                except HttpResponseError as e:
                    self.logger.error(f"Lỗi HTTP khi truy vấn Workspace ID: {workspace_id}.")
                    if e.response:
                        self.logger.error(f"Chi tiết phản hồi API: {e.response.json()}")
                except Exception as e:
                    self.logger.error(f"Lỗi không xác định khi truy vấn Workspace ID: {workspace_id}. Lỗi: {e}")

        except Exception as e:
            self.logger.error(f"Lỗi chung khi lấy dữ liệu: {e}", exc_info=True)

        return results


    def analyze_traffic_anomalies(self, tables: List[Any]) -> None:
        """
        Phân tích dữ liệu để phát hiện hơn 100 kết nối mỗi phút đến IP/cổng.
        - tables: danh sách các bảng dữ liệu trả về từ Traffic Analytics
        """
        if not tables:
            return

        for table in tables:
            for row in table.rows:
                # Chuyển đổi row thành dict để xử lý dễ dàng
                row_dict = dict(zip(table.columns, row))
                destination_ip = row_dict.get("DestinationIP_s")
                destination_port = row_dict.get("DestinationPort")
                connection_count = row_dict.get("ConnectionCount", 0)

                # Kiểm tra xem có vượt ngưỡng 100 kết nối mỗi phút
                if connection_count > 100:
                    self.logger.warning(
                        f"Phát hiện bất thường: {connection_count} kết nối mỗi phút đến "
                        f"IP: {destination_ip}, cổng: {destination_port}"
                    )

    def enable_traffic_analytics(
        self,
        resource_group: str,
        network_watcher_name: str,
        nsg_name: str,
        workspace_resource_id: str,
        storage_account_id: str,
        retention_days: int = 7
    ) -> bool:
        """
        Bật Traffic Analytics (Flow Logs) cho NSG nếu chưa bật.
        Sau đó kiểm tra trạng thái provisioning để đảm bảo đã bật thành công.
        """
        try:
            # 1) Kiểm tra xem flow log đã tồn tại / đang bật hay chưa
            #    (Nếu chưa tồn tại, lời gọi .get(...) có thể ném lỗi, ta sẽ bắt trong except)
            existing_flow_log = None
            flow_log_name = f"{nsg_name}-flowlog"

            try:
                existing_flow_log = self.network_client.flow_logs.get(
                    resource_group_name=network_watcher_name,  # Nếu Network Watcher ở cùng RG
                    network_watcher_name=network_watcher_name,
                    flow_log_name=flow_log_name
                )
            except Exception:
                # Chưa tồn tại flow log, ta sẽ tạo mới
                pass

            # Nếu Flow Log đã tồn tại và provisioning_state == "Succeeded" thì bỏ qua
            if existing_flow_log and getattr(existing_flow_log, 'provisioning_state', None) == "Succeeded":
                self.logger.info(
                    f"Flow Log '{flow_log_name}' cho NSG '{nsg_name}' đã được bật (provisioning_state=Succeeded)."
                )
                return True

            # 2) Nếu chưa bật hoặc đang ở trạng thái khác, tiến hành bật
            self.logger.info(f"Tiến hành bật Traffic Analytics cho NSG '{nsg_name}'...")

            parameters = {
                "location": self.get_nsg_location(resource_group, nsg_name),
                "enabled": True,
                "storageId": storage_account_id,
                "retentionPolicy": {
                    "days": retention_days,
                    "enabled": True
                },
                "format": "JSON",
                "flowAnalyticsConfiguration": {
                    "networkWatcherFlowAnalyticsConfiguration": {
                        "enabled": True,
                        "workspaceId": workspace_resource_id,
                        "workspaceRegion": self.get_workspace_region(workspace_resource_id),
                        "trafficAnalyticsInterval": 10
                    }
                }
            }

            # Tạo hoặc cập nhật flow log
            poller = self.network_client.flow_logs.begin_create_or_update(
                resource_group_name=resource_group,
                network_security_group_name=nsg_name,
                flow_log_name=flow_log_name,
                parameters=parameters
            )
            poller.result()  # Chờ lệnh hoàn thành

            self.logger.info(
                f"Đã gửi yêu cầu bật Traffic Analytics cho NSG '{nsg_name}'. Kiểm tra trạng thái..."
            )

            # 3) Kiểm tra trạng thái provisioning
            check_status = self.check_flow_log_status(
                network_watcher_resource_group=network_watcher_name,
                network_watcher_name=network_watcher_name,
                flow_log_name=flow_log_name
            )

            if check_status and check_status.get("state") == "Succeeded":
                self.logger.info(
                    f"Flow Log '{flow_log_name}' cho NSG '{nsg_name}' đã ở trạng thái Succeeded."
                )
                return True
            else:
                current_state = check_status.get("state") if check_status else "Unknown"
                self.logger.warning(
                    f"Flow Log '{flow_log_name}' cho NSG '{nsg_name}' chưa ở trạng thái Succeeded (hiện: {current_state})."
                )
                return False

        except Exception as e:
            self.logger.error(f"Lỗi khi bật Traffic Analytics cho NSG '{nsg_name}': {e}", exc_info=True)
            return False

    def check_flow_log_status(
        self,
        network_watcher_resource_group: str,
        network_watcher_name: str,
        flow_log_name: str
    ) -> Optional[dict]:
        """
        Kiểm tra trạng thái của flow log. Trả về dict gồm:
        {
            "id": flow_log.id,
            "state": flow_log.provisioning_state,
            "storageId": flow_log.storage_id,
            "targetResourceId": flow_log.target_resource_id
        }
        Nếu lỗi, trả về None.
        """
        try:
            flow_log = self.network_client.flow_logs.get(
                resource_group_name=network_watcher_resource_group,
                network_watcher_name=network_watcher_name,
                flow_log_name=flow_log_name
            )
            return {
                "id": flow_log.id,
                "state": flow_log.provisioning_state,
                "storageId": flow_log.storage_id,
                "targetResourceId": flow_log.target_resource_id
            }
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra trạng thái Flow Log '{flow_log_name}': {e}")
            return None

    def get_nsg_location(self, resource_group: str, nsg_name: str) -> str:
        """
        Trả về location của NSG. Mặc định 'eastus' nếu có lỗi.
        """
        try:
            nsg = self.network_client.network_security_groups.get(resource_group, nsg_name)
            return nsg.location
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy vị trí của NSG '{nsg_name}': {e}")
            return "eastus"

    def get_workspace_region(self, workspace_resource_id: str) -> str:
        """
        Lấy location của Workspace qua ResourceManagementClient. 
        Mặc định trả về 'eastus' nếu lỗi.
        """
        try:
            workspace = self.resource_management_client.resources.get_by_id(
                workspace_resource_id,
                '2015-11-01-preview'
            )
            return workspace.location
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy khu vực của Workspace '{workspace_resource_id}': {e}")
            return "eastus"
        

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Thiết lập logger cơ bản.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_format = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(console_format)

    logger.addHandler(console_handler)
    return logger


def main():
    # Thiết lập logger
    logger = setup_logger("TestTrafficAnalytics")

    # Khởi tạo AzureTrafficAnalyticsClient
    try:
        azure_traffic_client = AzureTrafficAnalyticsClient(logger)
        logger.info("Đã khởi tạo AzureTrafficAnalyticsClient thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo AzureTrafficAnalyticsClient: {e}", exc_info=True)
        return

    # Lấy danh sách Workspace IDs
    try:
        workspace_ids = azure_traffic_client.get_traffic_workspace_ids()
        if workspace_ids:
            logger.info(f"Danh sách Workspace IDs: {workspace_ids}")
        else:
            logger.warning("Không tìm thấy Workspace nào.")
            return
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách Workspace: {e}", exc_info=True)
        return

    # Truy vấn dữ liệu Traffic Analytics
    try:
        query = """
        AzureNetworkAnalytics_CL
        | summarize ConnectionCount = count() by DestinationIP_s, DestinationPort, bin(TimeGenerated, 1m)
        | where ConnectionCount > 100
        """

        logger.info("Bắt đầu truy vấn Traffic Analytics...")
        # Không cần chỉ định timespan, lớp sẽ tự động xác định
        traffic_data = azure_traffic_client.get_traffic_data(query=query)

        if traffic_data:
            logger.info(f"Đã lấy được {len(traffic_data)} bảng dữ liệu từ Traffic Analytics.")

            # Log thêm chi tiết bảng đầu tiên
            first_table = traffic_data[0] if traffic_data else None
            if first_table:
                logger.info(f"Các cột trong bảng đầu tiên: {first_table.columns}")
                logger.info(f"Số dòng trong bảng đầu tiên: {len(first_table.rows)}")

            # Phân tích dữ liệu bất thường
            logger.info("Bắt đầu phân tích bất thường trong dữ liệu...")
            azure_traffic_client.analyze_traffic_anomalies(traffic_data)
        else:
            logger.warning("Không có dữ liệu trả về từ Traffic Analytics.")
    except Exception as e:
        logger.error(f"Lỗi khi truy vấn hoặc phân tích Traffic Analytics: {e}", exc_info=True)

                # 6) Kiểm tra traffic từ Azure Traffic Analytics
                        traffic_data = self.resource_manager.azure_traffic_analytics_client.get_traffic_data()

                        # Nếu dữ liệu trả về không rỗng, kiểm tra xem có bất thường hay không
                        if traffic_data:
                            anomalies_detected = False

                            # Phân tích từng bảng dữ liệu
                            for table in traffic_data:
                                for row in table.rows:
                                    row_dict = dict(zip(table.columns, row))
                                    destination_ip = row_dict.get("DestinationIP_s")
                                    destination_port = row_dict.get("DestinationPort")
                                    connection_count = row_dict.get("ConnectionCount", 0)

                                     # Nếu vượt ngưỡng 100 kết nối mỗi phút
                                    if connection_count > 100:
                                        self.logger.warning(
                                            f"Detected traffic anomalies: {connection_count} connections/min to "
                                            f"IP: {destination_ip}, Port: {destination_port} (PID: {process.pid})"
                                        )
                                        anomalies_detected = True
                            # Nếu có bất thường, xử lý cloaking process
                            if anomalies_detected:
                                self.resource_manager.cloaking_request_queue.put(process)
                                process.is_cloaked = True
                                continue
                        
            except Exception as e:
                self.logger.error(f"Error in anomaly_detection: {e}")
        
        
if __name__ == "__main__":
    main()
