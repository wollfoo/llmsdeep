# resource_discovery.py

import logging
import os
from azure.identity import DefaultAzureCredential
from azure.mgmt.resourcegraph import ResourceGraphClient
from azure.resourcegraph.models import QueryRequest
from cachetools import cached, TTLCache
from azure.core.exceptions import HttpResponseError

class ResourceDiscovery:
    """
    Lớp để phát hiện tài nguyên Azure động sử dụng Azure Resource Graph.
    """
    def __init__(self, credential=None, subscription_id=None, logger=None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.credential = credential or DefaultAzureCredential()
        self.subscription_id = subscription_id or os.getenv('AZURE_SUBSCRIPTION_ID')
        self.resource_graph_client = ResourceGraphClient(self.credential)
    
    @cached(cache=TTLCache(maxsize=1024, ttl=600))
    def discover_resources(self, resource_types=None):
        """
        Phát hiện và trả về danh sách các tài nguyên Azure dựa trên các loại tài nguyên cung cấp.
        Nếu không cung cấp resource_types, tất cả tài nguyên sẽ được trả về.

        Args:
            resource_types (list): Danh sách các loại tài nguyên để lọc.

        Returns:
            list: Danh sách các tài nguyên đã phát hiện.
        """
        try:
            if resource_types:
                # Xây dựng truy vấn động dựa trên resource_types
                type_filters = " or ".join([f"type =~ '{rtype}'" for rtype in resource_types])
                query = f"Resources | where {type_filters}"
            else:
                # Nếu không cung cấp resource_types, trả về tất cả tài nguyên
                query = "Resources"
            request = QueryRequest(
                subscriptions=[self.subscription_id],
                query=query
            )
            response = self.resource_graph_client.resources(request)
            resources = response.data
            self.logger.info(f"Discovered {len(resources)} resources.")
            return resources
        except HttpResponseError as e:
            self.logger.error(f"HTTP error discovering resources: {e.message}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error discovering resources: {e}")
            return []
    
    @cached(cache=TTLCache(maxsize=1024, ttl=600))
    def get_resource_ids_by_type(self, resource_type):
        """
        Lấy danh sách ID của các tài nguyên theo loại dịch vụ cụ thể.

        Args:
            resource_type (str): Loại tài nguyên để lọc.

        Returns:
            list: Danh sách ID tài nguyên.
        """
        try:
            query = f"Resources | where type =~ '{resource_type}' | project id"
            request = QueryRequest(
                subscriptions=[self.subscription_id],
                query=query
            )
            response = self.resource_graph_client.resources(request)
            resources = response.data
            resource_ids = [res['id'] for res in resources]
            self.logger.info(f"Found {len(resource_ids)} resources of type {resource_type}.")
            return resource_ids
        except HttpResponseError as e:
            self.logger.error(f"HTTP error fetching resources of type {resource_type}: {e.message}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching resources of type {resource_type}: {e}")
            return []
    
    @cached(cache=TTLCache(maxsize=1024, ttl=600))
    def get_resources_by_type(self, resource_type):
        """
        Lấy danh sách tài nguyên theo loại dịch vụ cụ thể.

        Args:
            resource_type (str): Loại tài nguyên để lọc.

        Returns:
            list: Danh sách tài nguyên.
        """
        try:
            query = f"Resources | where type =~ '{resource_type}'"
            request = QueryRequest(
                subscriptions=[self.subscription_id],
                query=query
            )
            response = self.resource_graph_client.resources(request)
            resources = response.data
            self.logger.info(f"Found {len(resources)} resources of type {resource_type}.")
            return resources
        except HttpResponseError as e:
            self.logger.error(f"HTTP error fetching resources of type {resource_type}: {e.message}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching resources of type {resource_type}: {e}")
            return []
