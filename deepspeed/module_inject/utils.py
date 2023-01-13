# helper function to map between DS policies and DS containers
def policy_to_ds_container(**kwargs):
    container = None
    policy = kwargs['policy']

    if policy is None:
        print("Policy is None")
    else:
        from .containers import HFGPT2LayerPolicy
        from .containers import HFBertLayerPolicy
        from .containers import BLOOMLayerPolicy
        from .containers import HFGPTJLayerPolicy
        from .containers import HFGPTNEOLayerPolicy
        from .containers import GPTNEOXLayerPolicy
        from .containers import HFOPTLayerPolicy
        from .containers import MegatronLayerPolicy
        from .containers import HFDistilBertLayerPolicy

        if isinstance(policy, HFGPT2LayerPolicy):
            from .containers import DS_GPT2Container
            container = DS_GPT2Container(**kwargs)
        elif isinstance(policy, HFBertLayerPolicy):
            from .containers import DS_BERTContainer
            container = DS_BERTContainer(**kwargs)
        elif isinstance(policy, BLOOMLayerPolicy):
            from .containers import DS_BloomContainer
            container = DS_BloomContainer(**kwargs)
        elif isinstance(policy, HFGPTJLayerPolicy):
            from .containers import DS_GPTJContainer
            container = DS_GPTJContainer(**kwargs)
        elif isinstance(policy, HFGPTNEOLayerPolicy):
            from .containers import DS_GPTNEOContainer
            container = DS_GPTNEOContainer(**kwargs)
        elif isinstance(policy, GPTNEOXLayerPolicy):
            from .containers import DS_GPTNEOXContainer
            container = DS_GPTNEOXContainer(**kwargs)
        elif isinstance(policy, HFOPTLayerPolicy):
            from .containers import DS_OPTContainer
            container = DS_OPTContainer(**kwargs)
        elif isinstance(policy, MegatronLayerPolicy):
            from .containers import DS_MegatronGPTContainer
            container = DS_MegatronGPTContainer(**kwargs)
        elif isinstance(policy, HFDistilBertLayerPolicy):
            from .containers import DS_DistilBERTContainer
            container = DS_DistilBERTContainer(**kwargs)
        else:
            print("policy file is not recognized")

    return container
