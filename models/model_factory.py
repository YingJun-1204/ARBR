from models import gs_linear

class ModelFactory:
    
    @classmethod
    def Model(cls, configs):
        variant = getattr(configs, 'model_variant', 'baseline')
        # print(f"[Model Factory] 路由已命中 -> 正在初始化: {variant} variant in unified gs model")
        return gs_linear.Model(configs)