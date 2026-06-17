import paramiko
from scp import SCPClient

def create_ssh_client(server: str, port: int, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def send_file_to_server(ssh_client: paramiko.SSHClient, local_file: str, remote_path: str) -> None:
    scp = SCPClient(ssh_client.get_transport())
    try:
        scp.put(local_file, remote_path)
    finally:
        scp.close()
        ssh_client.close()

