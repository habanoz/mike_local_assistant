import yaml


def load_yaml_file(file):
    with open(file) as stream:
        try:
            return yaml.safe_load(stream)
        except Exception:
            raise
