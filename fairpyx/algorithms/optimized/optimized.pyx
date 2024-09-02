#optimized.pyx
import threading

import logging

import networkx as nx
from itertools import cycle

from fairpyx import AllocationBuilder
from fairpyx.algorithms.heterogeneous_matroid_constraints_algorithms import helper_update_envy_graph, \
    helper_remove_cycles, helper_update_ordered_agent_list, helper_create_agent_item_bipartite_graph, \
    helper_update_item_list
from fairpyx.allocations import AllocationBuilder
from multiprocessing import Process


# import concurrent.futures
# WORKERS=8
# lock = threading.Lock()
logger = logging.getLogger()


# Assuming AllocationBuilder, logger, and related classes/functions are defined elsewhere
# You may need to provide Cython definitions or cimport them as needed.


from itertools import cycle
import logging

logger = logging.getLogger(__name__)

def helper_categorization_friendly_picking_sequence_optimized(
        alloc,
        agent_order: list,
        items_to_allocate: list,
        agent_category_capacities: dict[str, dict[str, int]],
        target_category: str,
        debug: bool = False
):
    """
    Optimized Round Robin algorithm respecting categorization.

    Allocates items to agents based on a round-robin approach within a specific category.

    :param alloc: AllocationBuilder instance representing the current allocation.
    :param agent_order: List specifying the order of agents.
    :param items_to_allocate: List of items to be allocated in this round.
    :param agent_category_capacities: Dict mapping agents to their capacities per category.
    :param target_category: The category on which to perform allocation.
    :param debug: Flag to control logging verbosity.
    """
    # Conditional logging based on debug flag
    def log_info(message):
        if debug:
            logger.info(message)

    # Validate input
    # helper_validate_duplicate(agent_order)
    # helper_validate_duplicate(items_to_allocate)
    # helper_validate_capacities(agent_category_capacities)

    if not isinstance(target_category, str):
        raise ValueError("target_category must be of type str!")

    # Extract all categories present in agent_category_capacities
    categories = set(
        category
        for capacities in agent_category_capacities.values()
        for category in capacities.keys()
    )

    log_info(f"Target category: {target_category}, Agent capacities: {agent_category_capacities}")

    if target_category not in categories:
        raise ValueError(f"Target category mistyped or not found: {target_category}")

    # Initialize agent order if not provided
    if not agent_order:
        agent_order = [
            agent for agent in alloc.remaining_agents()
            if agent_category_capacities.get(agent, {}).get(target_category, 0) > 0
        ]

    # Initialize remaining capacities and agents with capacity
    remaining_category_agent_capacities = {
        agent: agent_category_capacities[agent].get(target_category, 0)
        for agent in agent_order
    }

    # Convert items_to_allocate to a set for O(1) lookups and removals
    remaining_category_items = set(
        item for item in alloc.remaining_items() if item in items_to_allocate
    )

    log_info(
        f"Initial remaining items: {remaining_category_items}, "
        f"Agent capacities: {remaining_category_agent_capacities}"
    )
    log_info(f"Agent order: {agent_order}")

    # Initialize set of agents who can still receive items
    remaining_agents_with_capacities = {
        agent for agent, capacity in remaining_category_agent_capacities.items() if capacity > 0
    }

    # Precompute a cache for agent_item_values to avoid redundant computations
    agent_item_value_cache = {}

    # Define a helper function to get agent_item_value with caching
    def get_agent_item_value(agent, item):
        if agent not in agent_item_value_cache:
            agent_item_value_cache[agent] = {}
        if item not in agent_item_value_cache[agent]:
            agent_item_value_cache[agent][item] = alloc.instance.agent_item_value(agent, item)
        return agent_item_value_cache[agent][item]

    # Iterate over agents in a round-robin fashion
    for agent in cycle(agent_order):
        if not remaining_agents_with_capacities:
            log_info("No more agents with capacity. Exiting loop.")
            break  # Exit if no agents can receive more items

        # Skip agents without remaining capacity
        if remaining_category_agent_capacities.get(agent, 0) <= 0:
            remaining_agents_with_capacities.discard(agent)
            log_info(f"{agent} has no remaining capacity and is removed from consideration.")
            continue

        log_info(f"Processing agent: {agent}, Remaining capacity: {remaining_category_agent_capacities[agent]}")

        # Determine potential items for the agent (no duplicates)
        potential_items_for_agent = remaining_category_items - set(alloc.bundles.get(agent, []))
        log_info(f"Potential items for {agent}: {potential_items_for_agent}")

        if not potential_items_for_agent:
            # Agent has capacity but no eligible items to receive
            remaining_agents_with_capacities.discard(agent)
            log_info(
                f"{agent} has capacity but no eligible items. Removed from consideration."
            )
            continue

        # Select the best item based on agent's valuation
        best_item = max(
            potential_items_for_agent,
            key=lambda item: get_agent_item_value(agent, item)
        )
        log_info(f"Best item for {agent}: {best_item}")

        # Allocate the item to the agent
        alloc.give(agent, best_item)
        remaining_category_agent_capacities[agent] -= 1
        remaining_category_items.discard(best_item)  # Remove allocated item

        log_info(f"Allocated {best_item} to {agent}. Remaining capacity: {remaining_category_agent_capacities[agent]}")
        log_info(f"Remaining items: {remaining_category_items}")

        # Early exit if no items left
        if not remaining_category_items:
            log_info("All items have been allocated. Exiting loop.")
            break

    log_info(f"Final allocation: {alloc.bundles}")

def helper_priority_matching_optimized(agent_item_bipartite_graph, list current_order, alloc,
                                       dict remaining_category_agent_capacities):
    """
    Performs priority matching based on the agent-item bipartite graph and the current order of agents.
    """

    cdef set matching
    cdef tuple match
    cdef str agent, item

    if not isinstance(agent_item_bipartite_graph, nx.Graph):
        raise ValueError("agent_item_bipartite_graph must be of type nx.Graph.")

    matching = nx.max_weight_matching(agent_item_bipartite_graph)

    logger.info(f'matching is -> {matching}')

    for match in matching:
        if match[0] in current_order:  # Check if the first element of the match is in the current order
            if ((match[0], match[1]) not in alloc.remaining_conflicts) and match[0] in remaining_category_agent_capacities:
                alloc.give(match[0], match[1], logger)
                remaining_category_agent_capacities[match[0]] -= 1
                if remaining_category_agent_capacities[match[0]] <= 0:
                    del remaining_category_agent_capacities[match[0]]
        else:  # Otherwise, the match[1] must be in the current order
            if ((match[1], match[0]) not in alloc.remaining_conflicts) and match[1] in remaining_category_agent_capacities:
                alloc.give(match[1], match[0], logger)
                remaining_category_agent_capacities[match[1]] -= 1
                if remaining_category_agent_capacities[match[1]] <= 0:
                    del remaining_category_agent_capacities[match[1]]

def per_category_round_robin_optimized(alloc, item_categories: dict, agent_category_capacities: dict,
                                       initial_agent_order: list):
    envy_graph = nx.DiGraph()
    cdef list current_order = initial_agent_order
    valuation_func = alloc.instance.agent_item_value
    cdef str category

    for category in item_categories.keys():
        logger.info(f'\nCurrent category -> {category}')
        logger.info(f'Envy graph before RR -> {envy_graph.edges}')
        helper_categorization_friendly_picking_sequence_optimized(alloc, current_order, item_categories[category],
                                                        agent_category_capacities, category)
        envy_graph = helper_update_envy_graph(alloc.bundles, valuation_func, item_categories, agent_category_capacities)
        logger.info(f'Envy graph after RR -> {envy_graph.edges}')
        if not nx.is_directed_acyclic_graph(envy_graph):
            logger.info("Cycle removal started ")
            envy_graph = helper_remove_cycles(envy_graph, alloc, valuation_func, item_categories, agent_category_capacities)
            logger.info('Cycle removal ended successfully ')
        else:
            logger.info('no cycles detected yet')
        current_order = list(nx.topological_sort(envy_graph))
        logger.info(f"Topological sort -> {current_order} \n***************************** ")

    logger.info(f'alloc after termination of algorithm ->{alloc}')

def iterated_priority_matching_optimized(alloc, item_categories: dict, agent_category_capacities: dict):
    logger.info("Running Iterated Priority Matching")

    # Declare all variables at the beginning
    envy_graph = nx.DiGraph()
    cdef list current_order = list(alloc.remaining_agents())
    cdef valuation_func = alloc.instance.agent_item_value
    cdef str category
    cdef int maximum_capacity, i
    cdef dict remaining_category_agent_capacities
    cdef list remaining_category_items
    cdef list current_agent_list

    envy_graph.add_nodes_from(alloc.remaining_agents())

    for category in item_categories.keys():
        maximum_capacity = max(
            [agent_category_capacities[agent][category] for agent in agent_category_capacities.keys()])
        logger.info(f'\nCategory {category}, Th=max(kih) is -> {maximum_capacity}')

        remaining_category_agent_capacities = {
            agent: agent_category_capacities[agent][category] for agent in agent_category_capacities if
            agent_category_capacities[agent][category] != 0
        }

        remaining_category_items = [x for x in alloc.remaining_items() if x in item_categories[category]]
        current_agent_list = helper_update_ordered_agent_list(current_order, remaining_category_agent_capacities)

        logger.info(
            f'remaining_category_items before priority matching in category:{category}-> {remaining_category_items}')
        logger.info(f'current_agent_list before priority matching in category:{category} -> {current_agent_list}')

        for i in range(maximum_capacity):
            agent_item_bipartite_graph = helper_create_agent_item_bipartite_graph(
                agents=current_agent_list,
                items=[item for item in alloc.remaining_items() if item in item_categories[category]],
                valuation_func=valuation_func,
            )

            envy_graph = helper_update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func,
                                                  item_categories=item_categories,
                                                  agent_category_capacities=agent_category_capacities)
            topological_sort = list(nx.topological_sort(envy_graph))
            logger.info(f'topological sort is -> {topological_sort}')
            current_order = current_order if not topological_sort else topological_sort

            helper_priority_matching_optimized(agent_item_bipartite_graph, current_order, alloc,
                                               remaining_category_agent_capacities)
            logger.info(f'allocation after priority matching in category:{category} & i:{i} -> {alloc.bundles}')
            remaining_category_items = helper_update_item_list(alloc, category, item_categories)
            current_agent_list = helper_update_ordered_agent_list(current_order, remaining_category_agent_capacities)
            logger.info(
                f'current_item_list after priority matching in category:{category} & i:{i} -> {remaining_category_items}')
            logger.info(
                f'current_agent_list after priority matching in category:{category} & i:{i} -> {current_agent_list}')

        agents_with_remaining_capacities = [agent for agent, capacity in remaining_category_agent_capacities.items() if
                                            capacity > 0]
        logger.info(
            f'remaining_category_agent_capacities of agents capable of carrying arbitrary item ->{remaining_category_agent_capacities}')
        logger.info(
            f'Using round-robin to allocate the items that were not allocated in the priority matching ->{remaining_category_items}')

        helper_categorization_friendly_picking_sequence_optimized(alloc, agents_with_remaining_capacities,
                                                                  item_categories[category],
                                                                  agent_category_capacities={
                                                                      agent: {
                                                                          category: remaining_category_agent_capacities[
                                                                              agent]} for agent in
                                                                      remaining_category_agent_capacities.keys()},
                                                                  target_category=category)
    logger.info(f'FINAL ALLOCATION IS -> {alloc.bundles}')

def capped_round_robin_optimized(alloc, item_categories: dict, agent_category_capacities: dict,
                                 initial_agent_order: list, str target_category):
    """
    Optimized version of the capped_round_robin function.
    """

    cdef list current_order = initial_agent_order

    logger.info(f'Running Capped Round Robin. initial_agent_order -> {initial_agent_order}')

    # Call the optimized helper function for round robin allocation
    helper_categorization_friendly_picking_sequence_optimized(alloc, current_order, item_categories[target_category],
                                                              agent_category_capacities, target_category)
    logger.info(f'alloc after CRR -> {alloc.bundles}')

def two_categories_capped_round_robin_optimized(alloc, item_categories: dict, agent_category_capacities: dict,
                                      initial_agent_order: list, target_category_pair: tuple[str]):
    cdef list current_order = initial_agent_order
    logger.info(f'\nRunning two_categories_capped_round_robin, initial_agent_order -> {current_order}')
    logger.info(f'\nAllocating category {target_category_pair[0]}')
    helper_categorization_friendly_picking_sequence_optimized(alloc, current_order, item_categories[target_category_pair[0]],
                                                    agent_category_capacities, target_category=target_category_pair[0])
    logger.info(f'alloc after CRR#{target_category_pair[0]} ->{alloc.bundles}')
    current_order.reverse()  # reversing order
    logger.info(f'reversed initial_agent_order -> {current_order}')
    logger.info(f'\nAllocating category {target_category_pair[1]}')
    helper_categorization_friendly_picking_sequence_optimized(alloc, current_order, item_categories[target_category_pair[1]],
                                                    agent_category_capacities, target_category=target_category_pair[1])
    logger.info(f'alloc after CRR#{target_category_pair[1]} ->{alloc.bundles}')

def per_category_capped_round_robin_optimized(alloc, item_categories: dict,
                                                  agent_category_capacities: dict,
                                                  initial_agent_order: list):
        envy_graph = nx.DiGraph()
        cdef list current_order = initial_agent_order
        cdef  valuation_func = alloc.instance.agent_item_value
        cdef str category

        logger.info(f'Run Per-Category Capped Round Robin, initial_agent_order->{initial_agent_order}')

        for category in item_categories.keys():
            helper_categorization_friendly_picking_sequence_optimized(alloc=alloc, agent_order=current_order,
                                                                      items_to_allocate=item_categories[category],
                                                                      agent_category_capacities=agent_category_capacities,
                                                                      target_category=category)

            # Update envy graph only if necessary (consider optimization here)
            envy_graph = helper_update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func,
                                                  item_categories=item_categories,
                                                  agent_category_capacities=agent_category_capacities)
            current_order = list(nx.topological_sort(envy_graph))

            logger.info(
                f'alloc after RR in category ->{category} is ->{alloc.bundles}.\n Envy graph nodes->{envy_graph.nodes} edges->{envy_graph.edges}.\ntopological sort->{current_order}')

        logger.info(f'allocation after termination of algorithm -> {alloc.bundles}')

def two_categories_capped_round_robin_optimized_multi_process(alloc, item_categories: dict, agent_category_capacities: dict,
                                                initial_agent_order: list, target_category_pair: tuple[str]):
    logger.info(f'\nRunning two_categories_capped_round_robin, initial_agent_order -> {initial_agent_order}')

    def process_category(target_category, order):
        helper_categorization_friendly_picking_sequence_optimized(
            alloc, order, item_categories[target_category],
            agent_category_capacities, target_category=target_category
        )
        logger.info(f'alloc after CRR#{target_category} ->{alloc.bundles}')

    # Create a process for each category
    p1 = Process(target=process_category, args=(target_category_pair[0], initial_agent_order))
    p2 = Process(target=process_category, args=(target_category_pair[1], list(reversed(initial_agent_order))))

    # Start both processes
    p1.start()
    p2.start()

    # Wait for both processes to complete
    p1.join()
    p2.join()

import threading

# def two_categories_capped_round_robin_optimized_threads(alloc, item_categories: dict, agent_category_capacities: dict,
#                                                 initial_agent_order: list, target_category_pair: tuple[str]):
#     def allocate_category(category, order):
#         logger.info(f'\nAllocating category {category}')
#         helper_categorization_friendly_picking_sequence_optimized(alloc, order, item_categories[category],
#                                                                   agent_category_capacities, target_category=category)
#         logger.info(f'alloc after CRR#{category} ->{alloc.bundles}')
#
#     # Create threads for each category allocation
#     thread1 = threading.Thread(target=allocate_category, args=(target_category_pair[0], initial_agent_order))
#     thread2 = threading.Thread(target=allocate_category, args=(target_category_pair[1], list(reversed(initial_agent_order))))
#
#     # Start the threads
#     thread1.start()
#     thread2.start()
#
#     # Wait for both threads to complete
#     thread1.join()
#     thread2.join()
#

#source /home/abodi-massarwa/.virtualenvs/fairpyx1/bin/activate
