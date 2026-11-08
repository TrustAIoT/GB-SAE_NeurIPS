"""Explicit dataset selection shared by cached-feature experiment entries."""


def add_data_fields(schema):
    schema.add_field("imagenet_root", path="data.imagenet_root", type=str,
                     nullable=True, inherit=("base", "data.imagenet_root"))
    schema.add_field("activation_root", path="data.activation_root", type=str,
                     nullable=True, inherit=("base", "data.activation_root"))
    schema.add_field("split", path="data.split", type=str,
                     nullable=True, inherit=("base", "data.split"))


def apply_data_arguments(cfg, args):
    from gbsae.configuration import select_data
    selection = {key: getattr(args, key) for key in ("imagenet_root", "activation_root", "split")
                 if getattr(args, key, None) is not None}
    return select_data(cfg, selection)
