#optimized.pyx

from itertools import cycle
import logging

import networkx as nx

logger = logging.getLogger()

def helper_categorization_friendly_picking_sequence_optimized(object alloc, list agent_order, list items_to_allocate, dict agent_category_capacities, str target_category='c1'):
    cdef list remaining_category_items
    cdef dict remaining_category_agent_capacities
    cdef set remaining_agents_with_capacities
    cdef str agent
    cdef str best_item_for_agent
    if agent_order is None:
        agent_order = [agent for agent in alloc.remaining_agents() if agent_category_capacities[agent][target_category] > 0]

    remaining_category_agent_capacities = {agent: agent_category_capacities[agent][target_category] for agent in
                                           agent_category_capacities.keys()}
    logger.info(f"agent_category_capacities-> {agent_category_capacities}")
    remaining_category_items = [x for x in alloc.remaining_items() if x in items_to_allocate]
    logger.info(
        f'remaining_category_items -> {remaining_category_items} & remaining agent capacities {remaining_category_agent_capacities}')
    logger.info(f"Agent order is -> {agent_order}")
    remaining_agents_with_capacities = {agent for agent, capacity in remaining_category_agent_capacities.items() if
                                        capacity > 0}

    for agent in cycle(agent_order):
        logger.info("Looping agent %s, remaining capacity %s", agent, remaining_category_agent_capacities[agent])
        if remaining_category_agent_capacities[agent] <= 0:
            remaining_agents_with_capacities.discard(agent)
            if len(remaining_agents_with_capacities) == 0:
                logger.info(f'No more agents with capacity')
                break
            continue

        potential_items_for_agent = set(remaining_category_items).difference(alloc.bundles[agent])
        if len(potential_items_for_agent) == 0:
            logger.info(f'No potential items for agent {agent}')
            if agent in remaining_agents_with_capacities:
                remaining_agents_with_capacities.discard(agent)
                if len(remaining_agents_with_capacities) == 0:
                    logger.info(f'No more agents with capacity')
                    break
                continue

        best_item_for_agent = max(potential_items_for_agent, key=lambda item: alloc.instance.agent_item_value(agent, item))
        logger.info(f'picked best item for {agent} -> item -> {best_item_for_agent}')
        alloc.give(agent, best_item_for_agent)
        remaining_category_agent_capacities[agent] -= 1
        remaining_category_items = [x for x in alloc.remaining_items() if x in items_to_allocate]
        if len(remaining_category_items) == 0:
            logger.info(f'No more items in category')
            break
        logger.info(
            f'remaining_category_items -> {remaining_category_items} & remaining agents {remaining_category_agent_capacities}')

def helper_priority_matching_optimized(agent_item_bipartite_graph, list current_order, alloc,
                             dict remaining_category_agent_capacities):
    """
    Performs priority matching based on the agent-item bipartite graph and the current order of agents.

    :param agent_item_bipartite_graph: A bipartite graph with agents and items as nodes, and edges with weights representing preferences.
    :param current_order: The current order of agents for matching.
    :param alloc: An AllocationBuilder instance to manage the allocation process.
    :param remaining_category_agent_capacities: A dictionary mapping agents to their remaining capacities for the category.
    """
    cdef set matching = nx.max_weight_matching(agent_item_bipartite_graph)
    cdef tuple match
    cdef str agent, item

    logger.info(f'matching is -> {matching}')

    for match in matching:
        if match[0] in current_order:  # means agent and not item
            agent, item = match[0], match[1]
        else:  # means match[1] contains agent name
            agent, item = match[1], match[0]

        if ((agent, item) not in alloc.remaining_conflicts) and agent in remaining_category_agent_capacities:
            alloc.give(agent, item, logger)
            remaining_category_agent_capacities[agent] -= 1
            if remaining_category_agent_capacities[agent] <= 0:
                del remaining_category_agent_capacities[agent]
