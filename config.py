class ConfigProperties:
    def __init__(self):
        self._model_folder_dic = None
        self._root_task_folder = None

    @property
    def model_folder_dic(self):
        return self._model_folder_dic

    @model_folder_dic.setter
    def model_folder_dic(self, value):
        self._model_folder_dic = value
    
    @property
    def root_task_folder(self):
        return self._root_task_folder

    @root_task_folder.setter
    def root_task_folder(self, value):
        self._root_task_folder = value

# Create a global configuration object
config_properties = ConfigProperties()
