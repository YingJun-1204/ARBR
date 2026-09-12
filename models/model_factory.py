from models import arbd


class ModelFactory:
    @classmethod
    def Model(cls, configs):
        return arbd.Model(configs)
