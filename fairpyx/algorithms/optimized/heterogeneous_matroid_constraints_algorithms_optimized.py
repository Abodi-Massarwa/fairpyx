"""
An implementation of the algorithms in:
"Fair Division under Heterogeneous Matroid Constraints", by Dror, Feldman, Segal-Halevi (2020), https://arxiv.org/abs/2010.07280v4
Programmer: Abed El-Kareem Massarwa.
Date: 2024-03.
"""
import math
import random
import threading
from time import perf_counter
from itertools import cycle
from tests.test_heterogeneous_matroid_constraints_algorithms import *
import experiments_csv
from networkx import DiGraph
import fairpyx.algorithms
from fairpyx import Instance, AllocationBuilder
from fairpyx.algorithms import *
from fairpyx import divide
import networkx as nx
import matplotlib.pyplot as plt
import logging
from fairpyx.algorithms.optimized.optimized import helper_categorization_friendly_picking_sequence_optimized, helper_priority_matching_optimized
import time, math
import concurrent.futures
from fairpyx.algorithms.heterogeneous_matroid_constraints_algorithms import *
WORKERS=8
lock = threading.Lock()
logger = logging.getLogger(__name__)

def per_category_round_robin_optimized(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                                       initial_agent_order: list):
    """
    this is the Algorithm 1 from the paper
    per category round-robin is an allocation algorithm which guarantees EF1 (envy-freeness up to 1 good) allocation
    under settings in which agent-capacities are equal across all agents,
    no capacity-inequalities are allowed since this algorithm doesnt provie a cycle-prevention mechanism
    TLDR: same partition constriants , same capacities , may have different valuations across agents  -> EF1 allocation

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
    :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
    :param agent_category_capacities:  a dictionary of dictionaru in which in the first dimension we have agents then
    paired with a dictionary of category-capacity.
    :param initial_agent_order: a list representing the order we start with in the algorithm

    >>> # Example 1
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2']
    >>> items=['m1','m2','m3']
    >>> item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3']}
    >>> agent_category_capacities = {'Agent1': {'c1': 2, 'c2': 2}, 'Agent2': {'c1': 2, 'c2': 2}}
    >>> valuations = {'Agent1':{'m1':2,'m2':8,'m3':7},'Agent2':{'m1':2,'m2':8,'m3':1}}
    >>> sum_agent_category_capacities={agent:sum(cap.values()) for agent,cap in agent_category_capacities.items()}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items,agent_capacities=sum_agent_category_capacities),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order)
    {'Agent1': ['m1', 'm3'], 'Agent2': ['m2']}

    >>> # Example 2
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent3','Agent2']
    >>> items=['m1','m2','m3']
    >>> item_categories = {'c1': ['m1','m3'], 'c2': ['m2']}
    >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':3}, 'Agent2': {'c1':3,'c2':3},'Agent3': {'c1':3,'c2':3}}
    >>> valuations = {'Agent1':{'m1':5,'m2':6,'m3':4},'Agent2':{'m1':6,'m2':5,'m3':6},'Agent3':{'m1':4,'m2':6,'m3':5}}
    >>> sum_agent_category_capacities={agent:sum(cap.values()) for agent,cap in agent_category_capacities.items()}
    >>> result=divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items,agent_capacities=sum_agent_category_capacities),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order)
    >>> assert result in [{'Agent1': ['m2'], 'Agent2': ['m1'], 'Agent3': ['m3']},{'Agent1': ['m1'], 'Agent2': ['m3'], 'Agent3': ['m2']}]

    >>> # example 3 but trying to get the expected output exactly (modified valuations different than on papers)  (4 agents ,4 items)
    >>> from fairpyx import  divide
    >>> order=['Agent1','Agent2','Agent3','Agent4']
    >>> items=['m1','m2','m3','m4']
    >>> item_categories = {'c1': ['m1', 'm2','m3'],'c2':['m4']}
    >>> agent_category_capacities = {'Agent1': {'c1':3,'c2':2}, 'Agent2': {'c1':3,'c2':2},'Agent3': {'c1':3,'c2':2},'Agent4': {'c1':3,'c2':2}} # in the papers its written capacity=size(catergory)
    >>> valuations = {'Agent1':{'m1':2,'m2':1,'m3':1,'m4':10},'Agent2':{'m1':1,'m2':2,'m3':1,'m4':10},'Agent3':{'m1':1,'m2':1,'m3':2,'m4':10},'Agent4':{'m1':1,'m2':1,'m3':1,'m4':10}}
    >>> sum_agent_category_capacities={agent:sum(cap.values()) for agent,cap in agent_category_capacities.items()}
    >>> divide(algorithm=per_category_round_robin,instance=Instance(valuations=valuations,items=items,agent_capacities=sum_agent_category_capacities),item_categories=item_categories,agent_category_capacities= agent_category_capacities,initial_agent_order=order)
    {'Agent1': ['m1'], 'Agent2': ['m2'], 'Agent3': ['m3'], 'Agent4': ['m4']}
    """
    logger.info(f"Running per_category_round_robin with alloc -> {alloc.bundles} \n item_categories -> {item_categories} \n agent_category_capacities -> {agent_category_capacities} \n -> initial_agent_order are -> {initial_agent_order}\n ")
    envy_graph = nx.DiGraph()
    current_order = initial_agent_order
    valuation_func = alloc.instance.agent_item_value

    for category in item_categories.keys():
        logger.info(f'\nCurrent category -> {category}')
        logger.info(f'Envy graph before RR -> {envy_graph.nodes}, edges -> in {envy_graph.edges}')
        helper_categorization_friendly_picking_sequence(alloc, current_order, item_categories[category], agent_category_capacities, category)
        helper_update_envy_graph(alloc.bundles, valuation_func, envy_graph, item_categories, agent_category_capacities)
        logger.info(f'Envy graph after  RR -> {envy_graph.nodes}, edges -> in {envy_graph.edges}')
        if not nx.is_directed_acyclic_graph(envy_graph):
            logger.info("Cycle removal started ")
            helper_remove_cycles(envy_graph, alloc, valuation_func, item_categories, agent_category_capacities)
            logger.info('cycle removal ended successfully ')
        current_order = list(nx.topological_sort(envy_graph))
        logger.info(f"Topological sort -> {current_order} \n***************************** ")
    logger.info(f'alloc after termination of algorithm ->{alloc}')


def capped_round_robin_optimized(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                       initial_agent_order: list, target_category: str):


    # no need for envy graphs whatsoever
    current_order = initial_agent_order
    logger.info(f'Running Capped Round Robin.  initial_agent_order -> {initial_agent_order}')
    helper_categorization_friendly_picking_sequence_optimized(alloc, current_order, item_categories[target_category], agent_category_capacities,
                                                    target_category=target_category)  # this is RR without wrapper
    logger.info(f'alloc after CRR -> {alloc.bundles}')

def two_categories_capped_round_robin_optimized(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                                      initial_agent_order: list, target_category_pair: tuple[str]):

    current_order = initial_agent_order
    logger.info(f'\nRunning two_categories_capped_round_robin, initial_agent_order -> {current_order}')
    logger.info(f'\nAllocating cagetory {target_category_pair[0]}')
    helper_categorization_friendly_picking_sequence_optimized(alloc, current_order, item_categories[target_category_pair[0]], agent_category_capacities,
                                                    target_category=target_category_pair[0])  #calling CRR on first category
    logger.info(f'alloc after CRR#{target_category_pair[0]} ->{alloc.bundles}')
    current_order.reverse()  #reversing order
    logger.info(f'reversed initial_agent_order -> {current_order}')
    logger.info(f'\nAllocating cagetory {target_category_pair[1]}')
    helper_categorization_friendly_picking_sequence_optimized(alloc, current_order, item_categories[target_category_pair[1]], agent_category_capacities,
                                                    target_category=target_category_pair[1])  # calling CRR on second category
    logger.info(f'alloc after CRR#{target_category_pair[1]} ->{alloc.bundles}')

def two_categories_capped_round_robin_optimized_threads(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                                      initial_agent_order: list, target_category_pair: tuple[str]):
    argument_list=[{'alloc':alloc,'item_categories':item_categories,'agent_category_capacities':agent_category_capacities,'initial_agent_order':initial_agent_order,'target_category':target_category_pair[0]},
    {'alloc':alloc,'item_categories':item_categories,'agent_category_capacities':agent_category_capacities,'initial_agent_order':list(reversed(initial_agent_order)),'target_category':target_category_pair[1]}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(capped_round_robin,**kwargs) for kwargs in argument_list]
        answers = []
        for future in concurrent.futures.as_completed(futures):  # return each result as soon as it is completed:
            answers.append(future.result())
def two_categories_capped_round_robin_optimized_threads_cython(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                                      initial_agent_order: list, target_category_pair: tuple[str]):
    argument_list=[{'alloc':alloc,'item_categories':item_categories,'agent_category_capacities':agent_category_capacities,'initial_agent_order':initial_agent_order,'target_category':target_category_pair[0]},
    {'alloc':alloc,'item_categories':item_categories,'agent_category_capacities':agent_category_capacities,'initial_agent_order':list(reversed(initial_agent_order)),'target_category':target_category_pair[1]}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(capped_round_robin_optimized,**kwargs) for kwargs in argument_list]
        answers = []
        for future in concurrent.futures.as_completed(futures):  # return each result as soon as it is completed:
            answers.append(future.result())

def per_category_capped_round_robin_optimized(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict,
                                    initial_agent_order: list):

    envy_graph = nx.DiGraph()
    current_order = initial_agent_order
    valuation_func = alloc.instance.agent_item_value
    logger.info(f'Run Per-Category Capped Round Robin, initial_agent_order->{initial_agent_order}')
    for category in item_categories.keys():
        helper_categorization_friendly_picking_sequence_optimized(alloc=alloc, agent_order=current_order,
                                           items_to_allocate=item_categories[category],
                                           agent_category_capacities=agent_category_capacities,
                                           target_category=category)
        helper_update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func, envy_graph=envy_graph,
                                 item_categories=item_categories, agent_category_capacities=agent_category_capacities)
        current_order = list(nx.topological_sort(envy_graph))
        logger.info(f'alloc after RR in category ->{category} is ->{alloc.bundles}.\n Envy graph nodes->{envy_graph.nodes} edges->{envy_graph.edges}.\ntopological sort->{current_order}')
    logger.info(f'allocation after termination of algorithm4 -> {alloc.bundles}')


def iterated_priority_matching_optimized(alloc: AllocationBuilder, item_categories: dict, agent_category_capacities: dict):
    """
    this is Algorithm 5  deals with (partition Matroids with Binary Valuations, may have different capacities)
    loops as much as maximum capacity in per each category , each iteration we build :
    1) agent-item graph (bidirectional graph)
    2) envy graph
    3) topological sort the order based on the envy graph (always a-cyclic under such settings,proven in papers)
    4) compute priority matching based on it we allocate the items among the agents
    we do this each loop , and in case there remains item in that category we arbitrarily give it to random agent

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents.
        :param item_categories: a dictionary of the categories  in which each category is paired with a list of items.
        :param agent_category_capacities:  a dictionary of dictionary in which in the first dimension we have agents then
        paired with a dictionary of category-capacity.
        :param order: a list representing the order we start with in the algorithm

            >>> # Example 1 (basic: 2 agents 3 items same capacities same valuations)
            >>> from fairpyx import  divide
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1','m2','m3']}
            >>> agent_category_capacities = {'Agent1': {'c1':1}, 'Agent2': {'c1':2}}
            >>> valuations = {'Agent1':{'m1':1,'m2':0,'m3':0},'Agent2':{'m1':0,'m2':1,'m3':0}}
            >>> divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
            {'Agent1': ['m1'], 'Agent2': ['m2', 'm3']}


            >>> # Example 2 ( 3 agents  with common interests in certain items)
            >>> from fairpyx import  divide
            >>> items=['m1','m2','m3']
            >>> item_categories = {'c1': ['m1'],'c2':['m2','m3']}
            >>> agent_category_capacities = {'Agent1': {'c1':2,'c2':2}, 'Agent2': {'c1':2,'c2':2},'Agent3': {'c1':2,'c2':2}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':1},'Agent2':{'m1':1,'m2':1,'m3':0},'Agent3':{'m1':0,'m2':0,'m3':0}} # TODO change valuation in paper
            >>> #divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)
            {'Agent1': ['m1', 'm3'], 'Agent2': ['m2'], 'Agent3': []}

            >>> # Example 3 ( 3 agents , 3 categories , with common interests, and remainder unallocated items at the end )
            >>> from fairpyx import  divide
            >>> items=['m1','m2','m3','m4','m5','m6']#TODO change in papers since in case there is no envy we cant choose whatever order we want. maybe on papers yes but in here no
            >>> item_categories = {'c1': ['m1','m2','m3'],'c2':['m4','m5'],'c3':['m6']}
            >>> agent_category_capacities = {'Agent1': {'c1':1,'c2':1,'c3':1}, 'Agent2': {'c1':1,'c2':1,'c3':1},'Agent3': {'c1':0,'c2':0,'c3':1}}
            >>> valuations = {'Agent1':{'m1':1,'m2':1,'m3':0,'m4':1,'m5':1,'m6':1},'Agent2':{'m1':0,'m2':1,'m3':0,'m4':1,'m5':1,'m6':1},'Agent3':{'m1':0,'m2':0,'m3':0,'m4':0,'m5':0,'m6':1}}
            >>> #divide(algorithm=iterated_priority_matching,instance=Instance(valuations=valuations,items=items),item_categories=item_categories,agent_category_capacities= agent_category_capacities)# m3 remains unallocated ....
            {'Agent1': ['m1', 'm5', 'm6'], 'Agent2': ['m2', 'm4'], 'Agent3': []}
   """
    logger.info("Running Iterated Priority Matching")
    envy_graph = nx.DiGraph()
    envy_graph.add_nodes_from(alloc.remaining_agents())  # adding agent nodes (no edges involved yet)
    current_order = list(alloc.remaining_agents())  # in this algorithm no need for initial_agent_order
    valuation_func = alloc.instance.agent_item_value

    for category in item_categories.keys():
        maximum_capacity = max(
            [agent_category_capacities[agent][category] for agent in
             agent_category_capacities.keys()])# for the sake of inner iteration
        logger.info(f'\nCategory {category}, Th=max(kih) is -> {maximum_capacity}')
        remaining_category_agent_capacities = {
            agent: agent_category_capacities[agent][category] for agent in agent_category_capacities if
            agent_category_capacities[agent][category] != 0
        }  # dictionary of the agents paired with capacities with respect to the current category we're dealing with

        # remaining_category_items = helper_update_item_list(alloc, category, item_categories)  # items we're dealing with with respect to the category
        remaining_category_items = [x for x in alloc.remaining_items() if x in item_categories[category]]
        current_agent_list = helper_update_ordered_agent_list(current_order, remaining_category_agent_capacities)  #  items we're dealing with with respect to the constraints
        logger.info(f'remaining_category_items before priority matching in category:{category}-> {remaining_category_items}')
        logger.info(f'current_agent_list before priority matching in category:{category} -> {current_agent_list}')
        for i in range(maximum_capacity):  # as in papers we run for the length of the maximum capacity out of all agents for the current category
            # Creation of agent-item graph
            agent_item_bipartite_graph = helper_create_agent_item_bipartite_graph(
                agents=current_agent_list,  # remaining agents
                items=[item for item in alloc.remaining_items() if item in item_categories[category]],
                # remaining items
                valuation_func=valuation_func,
              # remaining agents with respect to the order
            )  # building the Bi-Partite graph

            # Creation of envy graph
            helper_update_envy_graph(curr_bundles=alloc.bundles, valuation_func=valuation_func, envy_graph=envy_graph,
                                     item_categories=item_categories,
                                     agent_category_capacities=agent_category_capacities)  # updating envy graph with respect to matchings (first iteration we get no envy, cause there is no matching)
            #topological sort (papers prove graph is always a-cyclic)
            topological_sort = list(nx.topological_sort(envy_graph))
            logger.info(f'topological sort is -> {topological_sort}')
            current_order = current_order if not topological_sort else topological_sort
            # Perform priority matching
            helper_priority_matching_optimized(agent_item_bipartite_graph, current_order, alloc,
                                     remaining_category_agent_capacities)  # deals with eliminating finished agents from agent_category_capacities
            logger.info(f'allocation after priority matching in category:{category} & i:{i} -> {alloc.bundles}')
            remaining_category_items = helper_update_item_list(alloc, category,
                                                        item_categories)  # important to update the item list after priority matching.
            current_agent_list = helper_update_ordered_agent_list(current_order,
                                                                  remaining_category_agent_capacities)  # important to update the item list after priority matching.
            logger.info(f'current_item_list after priority matching in category:{category} & i:{i} -> {remaining_category_items}')
            logger.info(f'current_agent_list after priority matching in category:{category} & i:{i} -> {current_agent_list}')

        agents_with_remaining_capacities = [agent for agent,capacity in remaining_category_agent_capacities.items() if capacity>0]
        logger.info(f'remaining_category_agent_capacities of agents capable of carrying arbitrary item ->{remaining_category_agent_capacities}')
        logger.info(f'Using round-robin to allocate the items that were not allocated in the priority matching ->{remaining_category_items}')
        helper_categorization_friendly_picking_sequence(alloc, agents_with_remaining_capacities, item_categories[category], agent_category_capacities={agent:{category:remaining_category_agent_capacities[agent]} for agent in remaining_category_agent_capacities.keys()}, target_category=category)
    logger.info(f'FINAL ALLOCATION IS -> {alloc.bundles}')

if __name__ == "__main__":
    #logger=logging.getLogger()
    # logger = logging.getLogger()
    #
    # # Remove all handlers associated with the root logger
    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)
    #
    # # Set up your specific logger configuration
    # logger.setLevel(logging.INFO)
    #
    # # Add a FileHandler
    # file_handler = logging.FileHandler(filename='log_optimized', mode='w')
    # logger.addHandler(file_handler)


    start_time =perf_counter()
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(equal_capacities=True,
                                                                                           num_of_agents=50,
                                                                                           num_of_items=700,
                                                                                           category_count=20,
                                                                                           agent_capacity_bounds=(1,1),
                                                                                      random_seed_num=0)  # ,item_capacity_bounds=(1,1)#since we're doing cycle elemination
    # vals={agent:{item:instance.agent_item_value(agent, item) for item in instance.items} for agent in instance.agents}
    # print(f'number of cats {len(categories)}')
    # logger.info(f'done creating example ! \nagent valuations ->>>{vals} \n agent item capacities -> {agent_category_capacities} \n categories {categories}  \n initial order {initial_agent_order}')



    print('\nalgorithm 1: per category RR')
    alloc = divide(algorithm=per_category_round_robin,
                   instance=instance,
                   item_categories=categories,
                   agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order)

    end_time =perf_counter()
    print(f'non optimized time taken is -> {end_time - start_time}') #non optimized time taken is -> 7.78226479300065

    start_time = perf_counter()
    alloc = divide(algorithm=per_category_round_robin_optimized,
                   instance=instance,
                   item_categories=categories,
                   agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order)

    end_time = perf_counter()

    print(f'cythonized RR time taken is -> {end_time-start_time}')#cythonized RR time taken is -> 6.805714709000313

    print('\nAlgorithm2: CRR')
    alloc = divide(algorithm=capped_round_robin,
                   instance=instance,
                   item_categories=categories,
                   agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order,target_category='c1')

    end_time = perf_counter()
    print(f'non optimized time taken is -> {end_time - start_time}')  # non optimized time taken is -> 6.735089870999218


    start_time = perf_counter()
    alloc = divide(algorithm=capped_round_robin_optimized,
                   instance=instance,
                   item_categories=categories,
                   agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order,target_category='c1')

    end_time = perf_counter()

    print(f'cythonized RR time taken is -> {end_time - start_time}')  # cythonized RR time taken is -> 0.010826990997884423

    print('\nAlgorithm3: two categories CRR')
    alloc = divide(algorithm=two_categories_capped_round_robin,
                   instance=instance,
                   item_categories=categories,
                   agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order, target_category_pair=('c1','c2'))

    end_time = perf_counter()
    print(f'non optimized time taken is -> {end_time - start_time}')  # non optimized time taken is -> 0.13567857399903005

    start_time = perf_counter()
    alloc = divide(algorithm=two_categories_capped_round_robin_optimized,
                   instance=instance,
                   item_categories=categories,
                   agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order, target_category_pair=('c1','c2'))

    end_time = perf_counter()

    print(f'cythonized RR time taken is -> {end_time - start_time}')  # cythonized RR time taken is -> 0.05176478999783285
    start_time = perf_counter()
    alloc = divide(algorithm=two_categories_capped_round_robin_optimized_threads,
                   instance=instance,
                   item_categories=categories,
                   agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order, target_category_pair=('c1', 'c2'))

    end_time = perf_counter()

    print(
        f'threaded RR time taken is -> {end_time - start_time}')  #

    start_time = perf_counter()
    alloc = divide(algorithm=two_categories_capped_round_robin_optimized_threads_cython,
                   instance=instance,
                   item_categories=categories,
                   agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order, target_category_pair=('c1', 'c2'))

    end_time = perf_counter()

    print(
        f'cythonized&&THREADED RR time taken is -> {end_time - start_time}')  # cythonized RR time taken is -> 0.05176478999783285

    print('\nAlgorithm4: Per-category CRR')
    alloc = divide(algorithm=per_category_capped_round_robin,
                   instance=instance,
                   item_categories=categories,
                   agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order)

    end_time = perf_counter()
    print(f'non optimized time taken is -> {end_time - start_time}')  # non optimized time taken is -> 11.610839834000217

    start_time = perf_counter()
    alloc = divide(algorithm=per_category_capped_round_robin_optimized,
                   instance=instance,
                   item_categories=categories,
                   agent_category_capacities=agent_category_capacities,
                   initial_agent_order=initial_agent_order)

    end_time = perf_counter()

    print(
        f'cythonized RR time taken is -> {end_time - start_time}')  # cythonized RR time taken is -> 6.82228780999867

    print('\nAlgorithm5: Iterated Priority Matching')
    instance, agent_category_capacities, categories, initial_agent_order = random_instance(equal_capacities=False,
                                                                                           binary_valuations=True,
                                                                                           item_capacity_bounds=(1, 1),
                                                                                           random_seed_num=0)
    alloc = divide(algorithm=iterated_priority_matching,
                   instance=instance,
                   item_categories=categories, agent_category_capacities=agent_category_capacities)

    end_time = perf_counter()
    print(
        f'non optimized time taken is -> {end_time - start_time}')  # non optimized time taken is ->

    start_time = perf_counter()
    alloc = divide(algorithm=iterated_priority_matching_optimized,
                   instance=instance,
                   item_categories=categories, agent_category_capacities=agent_category_capacities)
    end_time = perf_counter()

    print(
        f'cythonized RR time taken is -> {end_time - start_time}')  # cythonized RR time taken is ->

"""
algorithm 1: per category RR
non optimized time taken is -> 15.133365936999326
cythonized RR time taken is -> 14.125902461000805

Algorithm2: CRR
non optimized time taken is -> 14.162945333000607
cythonized RR time taken is -> 0.03483409799991932

Algorithm3: two categories CRR
non optimized time taken is -> 0.11645089200010261
cythonized RR time taken is -> 0.08281217100011418
threaded RR time taken is -> 0.10131323399946268
cythonized&&THREADED RR time taken is -> 0.0841060570000991

Algorithm4: Per-category CRR
non optimized time taken is -> 14.253847054999824
cythonized RR time taken is -> 14.987228115000107

Algorithm5: Iterated Priority Matching
non optimized time taken is -> 15.009498536000137
cythonized RR time taken is -> 0.01908141700005217


"""