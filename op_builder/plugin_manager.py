import os

class PluginManager:
    def __init__(self):
        self.plugins = {}
        self.detect_plugins()

    def available_plugins(self):
        return list(self.plugins.keys())
    
    def detect_plugins(self):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../csrc/aio/plugins'))
        for device_type in os.listdir(base_path):
            plugin_dir = os.path.join(base_path, device_type)
            if os.path.isdir(plugin_dir):
                self.plugins[device_type] = {
                    'include_paths': self.get_include_paths(device_type),
                    'source_paths': self.get_sources(device_type)
                }

    def get_sources(self, device_type):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../csrc/aio/plugins', device_type))
        sources = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(('.cpp', '.h'))]
        return sources

    def get_include_paths(self, device_type):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../csrc/aio/plugins', device_type))
        include_paths = [base_path]
        return include_paths

    def get_plugin_info(self, device_type):
        if device_type in self.plugins:
            return self.plugins[device_type]
        else:
            return None

