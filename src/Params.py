import os
import sys
import json 
from collections import OrderedDict
import pdb


class Params:
    """
    @brief Parameter class
    """
    def __init__(self, filename=None):
        """
        @brief initialization
        """
        if filename is None or not os.path.exists(filename):
            filename = os.path.join(os.path.dirname(__file__), 'params.json')
        self.__dict__ = {}
        params_dict = {}
        with open(filename, "r") as f:
            params_dict = json.load(f, object_pairs_hook=OrderedDict)
        for key, value in params_dict.items():
            if 'default' in value: 
                self.__dict__[key] = value['default']
            else:
                self.__dict__[key] = None
        self.__dict__['params_dict'] = params_dict

    def printHelp(self):
        """
        @brief print help message for JSON parameters
        """
        content = self.toMarkdownTable()
        print(content)

    def toMarkdownTable(self):
        """
        @brief convert to markdown table 
        """
        key_length = len('JSON Parameter')
        key_length_map = []
        default_length = len('Default')
        default_length_map = []
        description_length = len('Description')
        description_length_map = []

        def getDefaultColumn(key, value):
            if sys.version_info.major < 3: # python 2
                flag = isinstance(value['default'], unicode)
            else: #python 3
                flag = isinstance(value['default'], str)
            if flag and not value['default'] and 'required' in value: 
                return value['required']
            else:
                return value['default']

        for key, value in self.params_dict.items():
            key_length_map.append(len(key))
            default_length_map.append(len(str(getDefaultColumn(key, value))))
            description_length_map.append(len(value['description']))
            key_length = max(key_length, key_length_map[-1])
            default_length = max(default_length, default_length_map[-1])
            description_length = max(description_length, description_length_map[-1])

        content = "| %s %s| %s %s| %s %s|\n" % (
                'JSON Parameter', 
                " " * (key_length - len('JSON Parameter') + 1), 
                'Default', 
                " " * (default_length - len('Default') + 1), 
                'Description', 
                " " * (description_length - len('Description') + 1)
                )
        content += "| %s | %s | %s |\n" % (
                "-" * (key_length + 1), 
                "-" * (default_length + 1), 
                "-" * (description_length + 1)
                )
        count = 0
        for key, value in self.params_dict.items():
            content += "| %s %s| %s %s| %s %s|\n" % (
                    key, 
                    " " * (key_length - key_length_map[count] + 1), 
                    str(getDefaultColumn(key, value)), 
                    " " * (default_length - default_length_map[count] + 1), 
                    value['description'], 
                    " " * (description_length - description_length_map[count] + 1)
                    )
            count += 1
        return content 

    def toJson(self):
        """
        @brief convert to json
        """
        data = {}
        for key, value in self.__dict__.items():
            if key != 'params_dict': 
                data[key] = value
        return data

    def fromJson(self, data):
        """
        @brief load form json
        """
        for key, value in data.items(): 
            self.__dict__[key] = value

    def dump(self, filename):
        """
        @brief dump to json file
        """
        with open(filename, 'w') as f:
            json.dump(self.toJson(), f)

    def load(self, filename):
        """
        @brief load from json file
        """
        with open(filename, 'r') as f:
            self.fromJson(json.load(f))

    def __str__(self):
        """
        @brief string representation with one key-value pair per line
        """
        data = self.toJson()
        lines = [f"{key}: {value}" for key, value in data.items()]
        return "\n".join(lines)

    def __repr__(self):
        """
        @brief print
        """
        return self.__str__()
