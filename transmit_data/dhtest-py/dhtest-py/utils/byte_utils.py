import struct

def get_length_prefix_bytes(input_bytes: bytes) -> bytes:
    data_length = len(input_bytes)
    length_prefix = struct.pack('!I', data_length)
    return length_prefix + input_bytes