"""Put the trips of a solution in the best order (highest raw profit first)."""


def sort_trips_by_profit(inst, solution):
    """Return a NEW solution with trips sorted by raw profit, highest first.

    The input is not modified. The sort is stable, so ties keep their order.
    """
    from uavplanner import profit_by_trip  # imported here to avoid circular imports

    raw = profit_by_trip(inst, solution)
    order = sorted(range(len(solution)), key=lambda k: (-raw[k], k))
    return [list(solution[k]) for k in order]