"""Scheduling for validation during OnePO training."""

def should_validate(step, interval, explicit_steps=(), final=False):
    """Combine periodic and explicit steps; preserve final-step validation."""
    steps = tuple(explicit_steps)
    if any(isinstance(s, bool) or not isinstance(s, int) or s <= 0 for s in steps):
        raise ValueError('TEST_STEPS must contain positive integers')
    enabled = interval > 0 or bool(steps)
    return enabled and (final or step in steps or (interval > 0 and step % interval == 0))
