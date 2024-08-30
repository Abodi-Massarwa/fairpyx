#optimized.pyx
import threading

import logging

import networkx as nx
from itertools import cycle
# import concurrent.futures
# WORKERS=8
# lock = threading.Lock()
logger = logging.getLogger()


# Assuming AllocationBuilder, logger, and related classes/functions are defined elsewhere
# You may need to provide Cython definitions or cimport them as needed.


def helper_categorization_friendly_picking_sequence_optimized(alloc, list agent_order, list items_to_allocate,
                                                              dict agent_category_capacities, str target_category):
    """
    This is a Round Robin algorithm with respect to categorization (works on each category separately when called).
    """

    cdef dict remaining_category_agent_capacities
    cdef list remaining_category_items
    cdef set remaining_agents_with_capacities
    cdef str agent, best_item_for_agent
    cdef int capacity

    # Replace lambda with a regular function
    def agent_item_value_key(item):
        return alloc.instance.agent_item_value(agent, item)

    # Validation
    # helper_validate_duplicate(agent_order)
    # helper_validate_duplicate(items_to_allocate)
    # helper_validate_capacities(agent_category_capacities)

    if not isinstance(target_category, str):
        raise ValueError("target_category must be of type str!")

    cdef list categories = list(
        set([category for agent, d in agent_category_capacities.items() for category in d.keys()]))
    logger.info(f"target category is ->{target_category}, agent_category_capacities are -> {agent_category_capacities}")

    if target_category not in categories:
        raise ValueError(f"Target category mistyped or not found: {target_category}")

    # Initialize variables
    if agent_order is None:
        agent_order = [agent for agent in alloc.remaining_agents() if
                       agent_category_capacities[agent][target_category] > 0]

    remaining_category_agent_capacities = {agent: agent_category_capacities[agent][target_category] for agent in
                                           agent_category_capacities.keys()}
    logger.info(f"agent_category_capacities -> {agent_category_capacities}")

    remaining_category_items = [x for x in alloc.remaining_items() if x in items_to_allocate]
    logger.info(
        f'remaining_category_items -> {remaining_category_items} & remaining agent capacities {remaining_category_agent_capacities}')
    logger.info(f"Agent order is -> {agent_order}")

    remaining_agents_with_capacities = {agent for agent, capacity in remaining_category_agent_capacities.items() if
                                        capacity > 0}

    # Round Robin Allocation Process
    for agent in cycle(agent_order):
        logger.info("Looping agent %s, remaining capacity %d", agent, remaining_category_agent_capacities[agent])

        if agent not in remaining_agents_with_capacities or not remaining_agents_with_capacities:
            if not remaining_agents_with_capacities:
                logger.info(
                    f'No agents left due to either:\n 1) reached maximum capacity\n 2) already has a copy of the item and can\'t carry more than one copy \n breaking out of loop!')
                break
            else:
                continue

        if remaining_category_agent_capacities[agent] <= 0:
            remaining_agents_with_capacities.discard(agent)
            logger.info(f'{agent} removed from loop since they have no capacity!')
            if len(remaining_agents_with_capacities) == 0:
                logger.info(f'No more agents with capacity')
                break
            continue

        potential_items_for_agent = set(remaining_category_items).difference(alloc.bundles[agent])
        logger.info(f'Potential set of items to be allocated to {agent} are -> {potential_items_for_agent}')

        if len(potential_items_for_agent) == 0:
            logger.info(f'No potential items for agent {agent}')
            logger.info(
                f'remaining_agents_with_capacities is -> {remaining_agents_with_capacities}, agent order is -> {agent_order}')
            if agent in remaining_agents_with_capacities:
                logger.info(f'{agent} still has capacity but already has a copy of the item')
                remaining_agents_with_capacities.discard(agent)
                logger.info(f'{agent} removed from loop')
                if len(remaining_agents_with_capacities) == 0:
                    logger.info(f'No more agents with capacity, breaking loop!')
                    break
                continue

        best_item_for_agent = max(potential_items_for_agent, key=agent_item_value_key)
        logger.info(f'Picked best item for {agent} -> item -> {best_item_for_agent}')
        alloc.give(agent, best_item_for_agent)
        remaining_category_agent_capacities[agent] -= 1

        remaining_category_items = [x for x in alloc.remaining_items() if x in items_to_allocate]
        if len(remaining_category_items) == 0:
            logger.info(f'No more items in category')
            break

        logger.info(
            f'remaining_category_items -> {remaining_category_items} & remaining agents {remaining_category_agent_capacities}')
# Assuming AllocationBuilder, logger, and related classes/functions are defined elsewhere
# You may need to provide Cython definitions or cimport them as needed.


def helper_priority_matching_optimized(agent_item_bipartite_graph, list current_order, alloc,
                                       dict remaining_category_agent_capacities):
    """
    Performs priority matching based on the agent-item bipartite graph and the current order of agents.
    """

    cdef set matching
    cdef tuple match
    cdef str agent, item

    # Validate input
    if not isinstance(agent_item_bipartite_graph, nx.Graph):
        raise ValueError("agent_item_bipartite_graph must be of type nx.Graph.")

    # helper_validate_duplicate(current_order)
    # helper_validate_capacities({'catx': remaining_category_agent_capacities})

    # Perform maximum weight matching
    matching = nx.max_weight_matching(agent_item_bipartite_graph)

    logger.info(f'matching is -> {matching}')

    for match in matching:
        if match[0] in current_order:  # Check if the first element of the match is in the current order
            if ((match[0], match[1]) not in alloc.remaining_conflicts) and match[
                0] in remaining_category_agent_capacities:
                alloc.give(match[0], match[1], logger)
                remaining_category_agent_capacities[match[0]] -= 1
                if remaining_category_agent_capacities[match[0]] <= 0:
                    del remaining_category_agent_capacities[match[0]]
        else:  # Otherwise, the match[1] must be in the current order
            if ((match[1], match[0]) not in alloc.remaining_conflicts) and match[
                1] in remaining_category_agent_capacities:
                alloc.give(match[1], match[0], logger)
                remaining_category_agent_capacities[match[1]] -= 1
                if remaining_category_agent_capacities[match[1]] <= 0:
                    del remaining_category_agent_capacities[match[1]]