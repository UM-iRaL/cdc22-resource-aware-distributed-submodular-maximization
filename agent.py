import matplotlib.patches as patch
import matplotlib.pyplot as plt
from observation_model import ObservationModel


class Agent(object):

    def __init__(self, state, radius, height, width, step, res, color):
        self.state = state
        self.radius = radius

        # Action Space
        self.actions = {'stay': (0, 0), 'up': (0, step), 'down': (0, -step),
                        'left': (-step, 0), 'right': (step, 0)}

        self.next_action = 'stay'
        self.n_actions = len(self.actions)


        # Observation Model
        self.observation_model = ObservationModel(int(radius), height, width, res)

        # Neighborhood
        self.in_neighbors = set()

        # Greedy selection
        self.gain = -1
        self.next_observed_points = set()
        self.unselected = 1  # whether i is already selected by RAG
        self.unchecked = 1   # whether i is already checked in an iteration (avoid multiple local maximum)

        # Plotting
        self.color = color
        self.patch = patch.Circle(xy=self.state, radius=self.radius, facecolor='none', edgecolor='k')#color=self.color
        # center = self.motion_model(self.state, [-self.radius, -self.radius])
        # self.patch = patch.Rectangle(xy=center, width=2*self.radius, height=2*self.radius, color=self.color)
        obs_points = self.observation_model.get_observed_points(self.state, return_all=True)
        # self.obs_patches = [patch.Rectangle(xy=point, width=0.3, height=0.3, color='k') for point in obs_points]
        self.obs_patches = [patch.Circle(xy=point, radius=0.05, color='k') for point in obs_points]

    def get_successors(self):
        """
        Returns possible subsequent states along each valid action, given the current state.
        :return: The list of subequent states.
        """
        return [(self.motion_model(self.state, action), name) for name, action in self.actions.items()]

    def get_observations(self, state):
        """
        Returns the observations at a potential new state.
        :param state: The state to observe from.
        :return: The set of observed points at the new state
        """
        return self.observation_model.get_observed_points(state)

    def set_next_action(self, action):
        """
        Assign next action.
        :param action: The action to assign
        :return: None
        """
        self.next_action = action

    def apply_next_action(self):
        """
        Applies the next action to modify the agent state.
        :return: None
        """
        self.state = self.motion_model(self.state, self.actions[self.next_action])
        self.update_patches()

    def update_patches(self):
        '''
        Updates the drawing of the field of view and observed points.
        :return: None
        '''
        # Update the Center of the Observable Square
        center = self.motion_model(self.state, [0, 0])
        self.patch.xy = center

        # Update the observed points drawing.
        obs_points = self.observation_model.get_observed_points(self.state)
        for obs_point, patch in zip(obs_points, self.obs_patches):
            patch.xy = obs_point

    def motion_model(self, state, action):
        '''
        Applies the motion model x_{t+1} = x_t + u_t, i.e. a discrete time control-additive motion model.
        :param state: The current state at time t.
        :param action: The current action at time t.
        :return: The resulting state x_{t+1}
        '''
        return state[0] + action[0], state[1] + action[1]
