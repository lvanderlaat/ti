""" Configuration

This module contains the class Conf which creates allow a dictionary to be
accesed by a dot. The function read takes a filepath to a YAML file and loads
it to a Conf file

"""
# Python Standard Library
import json

# Other dependencies
import yaml


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


class Conf(dict):
    def __init__(self, *args, **kwargs):
        super(Conf, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = Conf(v)
                    if isinstance(v, list):
                        self.__convert(v)
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    v = Conf(v)
                elif isinstance(v, list):
                    self.__convert(v)
                self[k] = v

    def __convert(self, v):
        for elem in range(0, len(v)):
            if isinstance(v[elem], dict):
                v[elem] = Conf(v[elem])
            elif isinstance(v[elem], list):
                self.__convert(v[elem])

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Conf, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Conf, self).__delitem__(key)
        del self.__dict__[key]


def read(filepath, filetype='yaml'):
    with open(filepath, 'r') as f:
        if filetype == 'yaml':
            return Conf(yaml.safe_load(f))
        elif filetype == 'json':
            return Conf(json.load(f))


if __name__ == '__main__':
    c = read_config('../config.yml')
    print(c.data.filepath); exit()
