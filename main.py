from agent import Agent
from simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
from planner import Planner
import time
import psutil
import networkx as nx

N_AGENTS = 5
WIDTH = 50
HEIGHT = 50
RADIUS = 10
COMM_RANGE = 80
STEP_SIZE = 3
RES = 1

def RAG():
    '''
    1st simulation in CDC'22
    random connected graph
    '''
    all_n_round = []   
    all_n_coverage = []  
    all_time = []  
    all_memory = []
    for rnd_seed in range(1, 2):
        np.random.seed(rnd_seed)
        agents, _ = connected_connectivity_graph()
        sim = Simulator(agents=agents, height=HEIGHT, width=WIDTH)
        planner = Planner()
        sim.draw()
        plt.savefig("map.png")

        start_time = time.time()
        n_coverage, n_round = planner.plan_rag(agents, COMM_RANGE)

        all_memory.append(np.round(psutil.Process().memory_info().rss / 1024 ** 2, 2))
        all_time.append(time.time() - start_time)
        all_n_round.append(n_round)
        all_n_coverage.append(n_coverage)
        
    print('Aver Comm Round is :', np.mean(all_n_round)) 
    print('Aver Coverage is :', np.mean(all_n_coverage)) 
    print('Aver Time is :', np.mean(all_time)) 
    print('Aver Memory is :', np.mean(all_memory))


def RAG_diff_comm_range():
    '''
    2nd simulation in CDC'22
    comparison of different communication ranges
    '''
    all_n_round = []   
    all_n_coverage = []  
    all_time = []  
    all_memory = []
    for communication_range in range(1, 51):
        np.random.seed(1)
        agents = [create_agent() for i in range(0, N_AGENTS)]
        sim = Simulator(agents=agents, height=HEIGHT, width=WIDTH)
        planner = Planner()

        start_time = time.time()

        n_coverage, n_round = planner.plan_rag(agents, communication_range)
        all_memory.append(np.round(psutil.Process().memory_info().rss / 1024 ** 2, 2))
        all_time.append(np.round(time.time() - start_time, 4))
        all_n_round.append(n_round)
        all_n_coverage.append(n_coverage)
        # all_memory.append(np.round(max(memory_usage((planner.plan_rag, (agents, communication_range))))), 3)
        
    print('Comm Round is :', all_n_round) 
    print('Coverage is :', all_n_coverage)
    print('Time is :', all_time)
    print('Memory is :', all_memory)

def create_agent():
    x = np.random.choice(range(0, HEIGHT))
    y = np.random.choice(range(0, WIDTH))
    return Agent(state=(x, y), radius=RADIUS, height=HEIGHT, width=WIDTH, step=STEP_SIZE, res=RES, color='none')

def create_sparse_agent():
    locations = [(10,10), (20,20), (20,40), (30,80), (40,50), (50,70), (60,40), (70,80), (80,30), (90,50)]
    agents = []
    for i in range(0,N_AGENTS):
        agents.append(Agent(state=locations[i], radius=RADIUS, height=HEIGHT, width=WIDTH, step=STEP_SIZE, res=RES, color=np.random.rand(3)))
    return agents

def connected_connectivity_graph():
    """
    Construct connected connectivity graph for agents.
    """
    agents = [create_agent() for i in range(0, N_AGENTS)]
    graph = connectivity_graph(agents, COMM_RANGE)
    while not nx.is_connected(graph):
        agents = [create_agent() for i in range(0, N_AGENTS)]
        graph = connectivity_graph(agents, COMM_RANGE)
    return agents, graph

def connectivity_graph(agents, comm_range):
    """
    Construct connectivity graph for agents.
    """
    G = nx.Graph()
    for idx_i, i in enumerate(agents):
        G.add_node(idx_i)
    for idx_i, i in enumerate(agents):
        for idx_j, j in enumerate(agents):
            if np.linalg.norm((i.state[0] - j.state[0], i.state[1] - j.state[1])) < comm_range \
            and i != j:
                G.add_edge(idx_i, idx_j)
    return G

if __name__ == "__main__":
    RAG()
    # RAG_diff_comm_range()
