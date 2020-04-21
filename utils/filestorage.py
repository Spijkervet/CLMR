import pathlib
from sacred.observers import FileStorageObserver

class CustomFileStorageObserver(FileStorageObserver):
    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        return super().started_event(ex_info, command, host_info, start_time, config, meta_info, _id)