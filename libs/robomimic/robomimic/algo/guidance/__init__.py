import types

from gr00t.model.policy import Gr00tPolicy


def modify_gr00t_policy(name: str, policy: Gr00tPolicy):
    print(f"\033[34m\033[1m[Modified GR00T-N1 policy to {name.upper()}.]\033[0m")
    name = name.lower()

    if name == 'no':
        pass
    ### ours ###
    elif name == "acg":
        from .acg import GR00T_N1_ACG, FlowmatchingActionHead_ACG, Gr00tPolicy_ACG

        policy.get_action = types.MethodType(Gr00tPolicy_ACG.get_action, policy)
        policy._get_action_from_normalized_input = types.MethodType(Gr00tPolicy_ACG._get_action_from_normalized_input, policy)
        policy.model.get_action = types.MethodType(GR00T_N1_ACG.get_action, policy.model)
        policy.model.action_head.get_action = types.MethodType(FlowmatchingActionHead_ACG.get_action, policy.model.action_head)

    ### our baseline methods ###
    elif name == "cfg":
        from .cfg import GR00T_N1_CFG, FlowmatchingActionHead_CFG, Gr00tPolicy_CFG

        policy.get_action = types.MethodType(Gr00tPolicy_CFG.get_action, policy)
        policy._get_action_from_normalized_input = types.MethodType(Gr00tPolicy_CFG._get_action_from_normalized_input, policy)
        policy.model.get_action = types.MethodType(GR00T_N1_CFG.get_action, policy.model)
        policy.model.action_head.get_action = types.MethodType(FlowmatchingActionHead_CFG.get_action, policy.model.action_head)
    elif name == "ensemble":
        from .ensemble import FlowmatchingActionHead_Ensemble, GR00T_N1_Ensemble, Gr00tPolicy_Ensemble

        policy.get_action = types.MethodType(Gr00tPolicy_Ensemble.get_action, policy)
        policy._get_action_from_normalized_input = types.MethodType(Gr00tPolicy_Ensemble._get_action_from_normalized_input, policy)
        policy.model.get_action = types.MethodType(GR00T_N1_Ensemble.get_action, policy.model)
        policy.model.action_head.get_action = types.MethodType(FlowmatchingActionHead_Ensemble.get_action, policy.model.action_head)
    elif name == "smooth":
        from .smooth import FlowmatchingActionHead_Smooth, GR00T_N1_Smooth, Gr00tPolicy_Smooth

        policy.get_action = types.MethodType(Gr00tPolicy_Smooth.get_action, policy)
        policy._get_action_from_normalized_input = types.MethodType(Gr00tPolicy_Smooth._get_action_from_normalized_input, policy)
        policy.model.get_action = types.MethodType(GR00T_N1_Smooth.get_action, policy.model)
        policy.model.action_head.get_action = types.MethodType(FlowmatchingActionHead_Smooth.get_action, policy.model.action_head)
    elif name == "smooth-feature":
        from .smooth_feature import FlowmatchingActionHead_Smooth_Feature, GR00T_N1_Smooth_Feature, Gr00tPolicy_Smooth_Feature

        policy.get_action = types.MethodType(Gr00tPolicy_Smooth_Feature.get_action, policy)
        policy._get_action_from_normalized_input = types.MethodType(Gr00tPolicy_Smooth_Feature._get_action_from_normalized_input, policy)
        policy.model.get_action = types.MethodType(GR00T_N1_Smooth_Feature.get_action, policy.model)
        policy.model.action_head.get_action = types.MethodType(FlowmatchingActionHead_Smooth_Feature.get_action, policy.model.action_head)
    elif name == "white-noise":
        from .white_noise import FlowmatchingActionHead_White_Noise, GR00T_N1_White_Noise, Gr00tPolicy_White_Noise

        policy.get_action = types.MethodType(Gr00tPolicy_White_Noise.get_action, policy)
        policy._get_action_from_normalized_input = types.MethodType(Gr00tPolicy_White_Noise._get_action_from_normalized_input, policy)
        policy.model.get_action = types.MethodType(GR00T_N1_White_Noise.get_action, policy.model)
        policy.model.action_head.get_action = types.MethodType(FlowmatchingActionHead_White_Noise.get_action, policy.model.action_head)
    else:
        raise NotImplementedError(f"Guidance type {name} is not supported.")

    return policy
