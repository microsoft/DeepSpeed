###############################################################
#           TODO: testing import of lib during runtime
###############################################################
#from .policies import HFGPT2LayerPolicy
#from .policies import HFBertLayerPolicy
#from .policies import BLOOMLayerPolicy
##import policies
##import containers
#
##policies = [HFGPT2LayerPolicy, HFBertLayerPolicy, BLOOMLayerPolicy]
##containers = [DS_GPT2Container, DS_BERTContainer, DS_BloomContainer]
##
##policy_container_map = dict(zip(policies, containers))
#
#policy_container_map = {
#    HFGPT2LayerPolicy : "DS_GPT2Container",
#    HFBertLayerPolicy : "DS_BERTContainer",
#    BLOOMLayerPolicy : "DS_BloomContainer"
#}
#
# TODO: String to class name
#import sys
#
#def str_to_class(classname, policy):
#    return getattr(sys.modules[__name__], classname)
#
#
## helper function to map between DS policies and DS containers
#def policy_to_ds_container(policy):
#    container = None
#
#    if policy is None:
#        print("Policy is None")
#    elif policy.__class__ in policy_container_map:
#    #else:
#        importlib = __import__('importlib')
#        #imported_container = importlib.import_module(".containers." + str(policy_container_map[policy.__class__]), package='.containers')
#
#        #imported_container = importlib.import_module(str(policy_container_map[policy.__class__]), package='.containers')
#        imported_container = importlib.import_module(str(policy_container_map[policy.__class__]), package='.containers.bloom')
#
#        #imported_container = __import__(".containers", fromlist=[str(policy_container_map[policy.__class__])], level=4)
#        container = imported_container(policy)
#
#    return container

#------------------------------------------------------------------------------------------------


# helper function to map between DS policies and DS containers
def policy_to_ds_container(policy, config, model_config, layer_id):
    container = None

    if policy is None:
        print("Policy is None")
    else:
        from .policies import HFGPT2LayerPolicy
        from .policies import HFBertLayerPolicy
        from .policies import BLOOMLayerPolicy
        from .policies import HFGPTJLayerPolicy
        from .policies import HFGPTNEOLayerPolicy
        from .policies import GPTNEOXLayerPolicy
        from .policies import HFOPTLayerPolicy
        from .policies import MegatronLayerPolicy
        from .policies import HFDistilBertLayerPolicy

        if isinstance(policy, HFGPT2LayerPolicy):
            print(f"policy is HFGPT2LayerPolicy")
            from .containers import DS_GPT2Container
            container = DS_GPT2Container(policy, config, model_config, layer_id)
        elif isinstance(policy, HFBertLayerPolicy):
            print("policy is HFBertLayerPolicy")
            from .containers import DS_BERTContainer
            container = DS_BERTContainer(policy, config, model_config, layer_id)
        elif isinstance(policy, BLOOMLayerPolicy):
            print("policy is BLOOMLayerPolicy")
            from .containers import DS_BloomContainer
            container = DS_BloomContainer(policy, config, model_config, layer_id)
        elif isinstance(policy, HFGPTJLayerPolicy):
            print(f"policy is HFGPTJLayerPolicy")
            from .containers import DS_GPTJContainer
            container = DS_GPTJContainer(policy, config, model_config, layer_id)
        elif isinstance(policy, HFGPTNEOLayerPolicy):
            print(f"policy is HFGPTNEOLayerPolicy")
            from .containers import DS_GPTNEOContainer
            container = DS_GPTNEOContainer(policy, config, model_config, layer_id)
        elif isinstance(policy, GPTNEOXLayerPolicy):
            print(f"policy is GPTNEOXLayerPolicy")
            from .containers import DS_GPTNEOXContainer
            container = DS_GPTNEOXContainer(policy, config, model_config, layer_id)
        elif isinstance(policy, HFOPTLayerPolicy):
            print(f"policy is HFOPTLayerPolicy")
            from .containers import DS_OPTContainer
            container = DS_OPTContainer(policy, config, model_config, layer_id)
        elif isinstance(policy, MegatronLayerPolicy):
            print(f"policy is MegatronLayerPolicy")
            from .containers import DS_MegatronGPTContainer
            container = DS_MegatronGPTContainer(policy, config, model_config, layer_id)
        elif isinstance(policy, HFDistilBertLayerPolicy):
            print(f"policy is HFDistilBertLayerPolicy")
            from .containers import DS_DistilBERTContainer
            container = DS_DistilBERTContainer(policy, config, model_config, layer_id)
        else:
            print("policy file is not recognized")

    return container
