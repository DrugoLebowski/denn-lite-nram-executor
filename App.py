# Project
from Singleton import Singleton


class App(metaclass=Singleton):

    _conf = {
        "images_path": "./images/"
    }

    @staticmethod
    def get(name):
        try:
            return App._conf[name]
        except KeyError:
            return None
