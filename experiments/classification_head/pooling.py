"""Mean token pooling used by the active classification protocols."""
def mean_pool_tokens(token_features):
    """Image-level features: arithmetic mean over spatial tokens."""
    return token_features.mean(dim=1)

