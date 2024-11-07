import os
import numpy as np
import networkx as nx
from bidict import bidict
from collections import OrderedDict

from scipy.sparse import coo_matrix

DLINK_ENUM = ['CTM', 'LQ', 'LTM', 'PQ']
DNODE_ENUM = ['FWJ', 'GRJ', 'DMOND', 'DMDND']
DNODE_COLOR_CODE = {"FWJ": "lightgrey", "GRJ": "#6DB1BF", "DMOND": "#F39A9D", "DMDND": "#3F6C51"}
DLINK_FORMAT_CODE = {"CTM": False, "LQ": True, "LTM": True, "PQ": True}
class MNM_dlink():
  """
  A class to represent a network link in a traffic simulation model.

  Attributes
  ----------
  ID : int
    The unique identifier for the link.
  length : float
    The length of the link in miles.
  typ : str
    The type of the link.
  ffs : float
    The free-flow speed of the link in miles per hour.
  cap : float
    The capacity of the link in vehicles per hour.
  rhoj : float
    The jam density of the link in vehicles per mile.
  lanes : int
    The number of lanes on the link.

  Methods
  -------
  get_fft():
    Calculates and returns the free-flow travel time of the link in seconds.
  is_ok(unit_time=5):
    Validates the attributes of the link.
  __str__():
    Returns a string representation of the link.
  __repr__():
    Returns a string representation of the link.
  generate_text():
    Generates a space-separated string of the link's attributes.
  """
  def __init__(self):
    self.ID = None
    self.length = None
    self.typ = None
    self.ffs = None
    self.cap = None
    self.rhoj = None
    self.lanes = None

  def __init__(self, ID, length, typ, ffs, cap, rhoj, lanes):
    self.ID = ID
    self.length = length  #mile
    self.typ = typ        #type
    self.ffs = ffs        #mile/h
    self.cap = cap        #v/hour
    self.rhoj = rhoj      #v/miles
    self.lanes = lanes    #num of lanes

  def get_fft(self):
    return self.length / self.ffs * 3600

  def is_ok(self, unit_time = 5):
    assert(self.length > 0.0)
    assert(self.ffs > 0.0)
    assert(self.cap > 0.0)
    assert(self.rhoj > 0.0)
    assert(self.lanes > 0)
    assert(self.typ in DLINK_ENUM)
    assert(self.cap / self.ffs < self.rhoj)
    # if self.ffs < 9999:
    #   assert(unit_time * self.ffs / 3600 <= self.length)

  def __str__(self):
    return "MNM_dlink, ID: {}, type: {}, length: {} miles, ffs: {} mi/h".format(self.ID, self.typ, self.length, self.ffs)

  def __repr__(self):
    return self.__str__()

  def generate_text(self):
    return ' '.join([str(e) for e in [self.ID, self.typ, self.length, self.ffs, self.cap, self.rhoj, self.lanes]])

class MNM_dnode():
  """
  Class representing a directed node (MNM_dnode).

  Attributes:
    ID (int or None): The identifier of the node.
    typ (str or None): The type of the node.

  Methods:
    __init__(ID, typ):
      Initializes a new instance of MNM_dnode with the given ID and type.
    is_ok():
      Checks if the node type is valid by asserting it is in DNODE_ENUM.
    __str__():
      Returns a string representation of the node.
    __repr__():
      Returns a string representation of the node (same as __str__()).
    generate_text():
      Generates a text representation of the node attributes.
  """
  def __init__(self):
    self.ID = None
    self.typ = None
  def __init__(self, ID, typ):
    self.ID = ID
    self.typ = typ
  def is_ok(self):
    assert(self.typ in DNODE_ENUM)

  def __str__(self):
    return "MNM_dnode, ID: {}, type: {}".format(self.ID, self.typ)

  def __repr__(self):
    return self.__str__()

  def generate_text(self):
    return ' '.join([str(e) for e in [self.ID, self.typ]])

class MNM_demand():
  """
  A class to represent and manage traffic demand data.

  Attributes
  ----------
  demand_dict : dict
    A dictionary to store demand data with origin (O) as keys and 
    another dictionary as values where destination (D) is the key 
    and demand is the value.
  demand_list : list
    A list to store tuples of (O, D) representing origin and 
    destination pairs.

  Methods
  -------
  add_demand(O, D, demand, overwriting=False):
    Adds demand data for a given origin (O) and destination (D). 
    Raises an error if the demand already exists and overwriting 
    is set to False.
  
  build_from_file(file_name):
    Reads demand data from a file and populates demand_dict and 
    demand_list attributes.
  
  __str__():
    Returns a string representation of the MNM_demand object 
    showing the number of origins and OD pairs.
  
  __repr__():
    Returns the string representation of the MNM_demand object.
  
  generate_text():
    Generates a text representation of the demand data in a 
    specific format.
  """
  def __init__(self):
    self.demand_dict = dict()
    self.demand_list = None

  def add_demand(self, O, D, demand, overwriting = False):
    assert(type(demand) is np.ndarray)
    assert(len(demand.shape) == 1)
    if O not in self.demand_dict.keys():
      self.demand_dict[O] = dict()
    if (not overwriting) and (D in self.demand_dict[O].keys()):
      raise("Error, exists OD demand already") 
    else:
      self.demand_dict[O][D] = demand

  def build_from_file(self, file_name):
    self.demand_list = list()
    # f = file(file_name)
    f = open(file_name, "r")
    log = f.readlines()[1:]
    f.close()
    for i in range(len(log)):
      tmp_str = log[i].strip()
      if tmp_str == '':
        continue
      words = tmp_str.split()
      O_ID = int(words[0])
      D_ID = int(words[1])
      demand = np.array(words[2:]).astype(float)
      self.add_demand(O_ID, D_ID, demand)
      self.demand_list.append((O_ID, D_ID))

  def __str__(self):
    return "MNM_demand, number of O: {}, number of OD: {}".format(len(self.demand_dict), len(self.demand_list))

  def __repr__(self):
    return self.__str__()

  def generate_text(self):
    tmp_str = '#Origin_ID Destination_ID <demand by interval>\n'
    for O in self.demand_dict.keys():
      for D in self.demand_dict[O].keys():
        tmp_str += ' '.join([str(e) for e in [O, D] + self.demand_dict[O][D].tolist()]) + '\n'
    return tmp_str


class MNM_od():
  """
  A class to represent the origin-destination (OD) mapping in a network.

  Attributes
  ----------
  O_dict : bidict
    A bidirectional dictionary to store origin nodes and their corresponding IDs.
  D_dict : bidict
    A bidirectional dictionary to store destination nodes and their corresponding IDs.

  Methods
  -------
  add_origin(O, Onode_ID, overwriting=False):
    Adds an origin node to the O_dict. Raises an error if the origin node already exists and overwriting is False.
  add_destination(D, Dnode_ID, overwriting=False):
    Adds a destination node to the D_dict. Raises an error if the destination node already exists and overwriting is False.
  build_from_file(file_name):
    Builds the O_dict and D_dict from a given file.
  generate_text():
    Generates a text representation of the O_dict and D_dict.
  __str__():
    Returns a string representation of the MNM_od object.
  __repr__():
    Returns a string representation of the MNM_od object.
  """
  def __init__(self):
    self.O_dict = bidict()
    self.D_dict = bidict()

  def add_origin(self, O, Onode_ID, overwriting = False):
    if (not overwriting) and (O in self.O_dict.keys()):
      raise("Error, exists origin node already")
    else:
      self.O_dict[O] = Onode_ID

  def add_destination(self, D, Dnode_ID, overwriting = False):
    if (not overwriting) and (D in self.D_dict.keys()):
      raise("Error, exists destination node already")
    else:
      self.D_dict[D] = Dnode_ID

  def build_from_file(self, file_name):
    self.O_dict = bidict()
    self.D_dict = bidict()
    flip = False
    # f = file(file_name)
    f = open(file_name, "r")
    log = f.readlines()[1:]
    f.close()
    for i in range(len(log)):
      tmp_str = log[i].strip()
      if tmp_str == '':
        continue
      if tmp_str.startswith('#'):
        flip = True
        continue
      words = tmp_str.split()
      od = int(words[0])
      node = int(words[1])
      if not flip:
        self.add_origin(od, node)
      else:
        self.add_destination(od, node)

  def generate_text(self):
    tmp_str = '#Origin_ID <-> node_ID\n'
    # python 2
    # for O_ID, node_ID in self.O_dict.iteritems():
    # python 3  
    for O_ID, node_ID in self.O_dict.items():
      tmp_str += ' '.join([str(e) for e in [O_ID, node_ID]]) + '\n'
    tmp_str += '#Dest_ID <-> node_ID\n'
    # python 2
    # for D_ID, node_ID in self.D_dict.iteritems():
    # python 3
    for D_ID, node_ID in self.D_dict.items():
      tmp_str += ' '.join([str(e) for e in [D_ID, node_ID]]) + '\n'
    return tmp_str

  def __str__(self):
    return "MNM_od, number of O:" + str(len(self.O_dict)) + ", number of D:" + str(len(self.D_dict))

  def __repr__(self):
    return self.__str__()

class MNM_graph():
  """
  A class to represent a directed graph using NetworkX.

  Attributes
  ----------
  G : nx.DiGraph
    A directed graph object from NetworkX.
  edgeID_dict : OrderedDict
    A dictionary to store edge IDs and their corresponding start and end nodes.

  Methods
  -------
  add_edge(s, e, ID, create_node=False, overwriting=False):
    Adds an edge to the graph.
  add_node(node, overwriting=True):
    Adds a node to the graph.
  build_from_file(file_name):
    Builds the graph from a file.
  __str__():
    Returns a string representation of the graph.
  __repr__():
    Returns a string representation of the graph.
  generate_text():
    Generates a text representation of the graph's edges.
  """
  def __init__(self):
    self.G = nx.DiGraph()
    self.edgeID_dict = OrderedDict()

  def add_edge(self, s, e, ID, create_node = False, overwriting = False):
    if (not overwriting) and ((s,e) in self.G.edges()):
      raise("Error, exists edge in graph")
    elif (not create_node) and s in self.G.nodes():
      raise("Error, exists start node of edge in graph")
    elif (not create_node) and e in self.G.nodes():
      raise("Error, exists end node of edge in graph")
    else:
      self.G.add_edge(s, e, ID = ID)
      self.edgeID_dict[ID] = (s, e)

  def add_node(self, node, overwriting = True):
    if (not overwriting) and node in self.G.nodes():
      raise("Error, exists node in graph")
    else:
      self.G.add_node(node)

  def build_from_file(self, file_name):
    self.G = nx.DiGraph()
    self.edgeID_dict = OrderedDict()
    # f = file(file_name)
    f = open(file_name, "r")
    log = f.readlines()[1:]
    f.close()
    for i in range(len(log)):
      tmp_str = log[i].strip()
      if tmp_str == '':
        continue
      words = tmp_str.split()
      assert(len(words) == 3)
      edge_id = int(words[0])
      from_id = int(words[1])
      to_id = int(words[2])
      self.add_edge(from_id, to_id, edge_id, create_node = True, overwriting = False)

  def __str__(self):
    return "MNM_graph, number of node:" + str(self.G.number_of_nodes()) + ", number of edges:" + str(self.G.number_of_edges())

  def __repr__(self):
    return self.__str__()

  def generate_text(self):
    tmp_str = '# EdgeId FromNodeId  ToNodeId\n'
    # python 2
    # for edge_id, (from_id, to_id) in self.edgeID_dict.iteritems():
    # python 3
    for edge_id, (from_id, to_id) in self.edgeID_dict.items():
      tmp_str += ' '.join([str(e) for e in [edge_id, from_id, to_id]]) + '\n'
    return tmp_str

class MNM_path():
  """
  MNM_path class represents a path in a network with a unique path ID, origin node, 
  destination node, and a list of nodes that form the path. It also includes route 
  portions for route choice modeling.
  Attributes:
    path_ID (int): Unique identifier for the path.
    origin_node (int): The starting node of the path.
    destination_node (int): The ending node of the path.
    node_list (list): List of nodes that form the path.
    route_portions (numpy.ndarray): Array representing route choice portions.
  Methods:
    __init__(node_list, path_ID):
      Initializes the MNM_path with a list of nodes and a path ID.
    __eq__(other):
      Checks if two MNM_path objects are equal based on their attributes.
    create_route_choice_portions(num_intervals):
      Creates an array of ones with the specified number of intervals for route portions.
    attach_route_choice_portions(portions):
      Attaches an array of route choice portions to the path.
    __ne__(other):
      Checks if two MNM_path objects are not equal.
    __str__():
      Returns a string representation of the MNM_path object.
    __repr__():
      Returns a string representation of the MNM_path object.
    generate_node_list_text():
      Generates a text representation of the node list.
    generate_portion_text():
      Generates a text representation of the route portions.
  """
  def __init__(self):
    # print "MNM_path"
    self.path_ID = None
    self.origin_node = None
    self.destination_node = None
    self.node_list = list()
    self.route_portions = None

  def __init__(self, node_list, path_ID):
    self.path_ID = path_ID
    self.origin_node = node_list[0]
    self.destination_node = node_list[-1]
    self.node_list = node_list
    self.route_portions = None

  def __eq__(self, other):
    if ((self.origin_node is None) or (self.destination_node is None) or 
         (other.origin_node is None) or (other.destination_node is None)):
      return False 
    if (other.origin_node != self.origin_node):
      return False
    if (other.destination_node != self.destination_node):
      return False   
    if (len(self.node_list) != len(other.node_list)):
      return False
    for i in range(len(self.node_list)):
      if self.node_list[i] != other.node_list[i]:
        return False
    return True

  def create_route_choice_portions(self, num_intervals):
    self.route_portions = np.ones(num_intervals)

  def attach_route_choice_portions(self, portions):
    self.route_portions = portions
  
  def __ne__(self, other):
    return not self.__eq__(other)

  def __str__(self):
    return "MNM_path, path ID {}, O node: {}, D node {}".format(self.path_ID, self.origin_node, self.destination_node)

  def __repr__(self):
    return self.__str__()

  def generate_node_list_text(self):
    return ' '.join([str(e) for e in self.node_list])

  def generate_portion_text(self):
    assert(self.route_portions is not None)
    return ' '.join(['{:.12f}'.format(e) for e in self.route_portions])

# This class is named MNM_pathset and likely contains functionality related to managing sets of paths.
class MNM_pathset():
  """
  A class to represent a set of paths between an origin and a destination node.

  Attributes
  ----------
  origin_node : None or node type
    The origin node of the path set.
  destination_node : None or node type
    The destination node of the path set.
  path_list : list
    A list to store paths in the path set.

  Methods
  -------
  add_path(path, overwriting=False):
    Adds a path to the path set. Raises an error if the path already exists and overwriting is False.
  normalize_route_portions(sum_to_OD=False):
    Normalizes the route portions of the paths in the path set. If sum_to_OD is True, returns the sum of the route portions.
  __str__():
    Returns a string representation of the path set.
  __repr__():
    Returns a string representation of the path set.
  """
  def __init__(self):
    self.origin_node = None
    self.destination_node = None
    self.path_list = list()

  def add_path(self, path, overwriting = False):
    assert(path.origin_node == self.origin_node)
    assert(path.destination_node == self.destination_node)
    if (not overwriting) and (path in self.path_list):
      raise ("Error, path in path set")
    else:
      self.path_list.append(path)

  def normalize_route_portions(self, sum_to_OD = False):
    for i in range(len(self.path_list) - 1):
      assert(self.path_list[i].route_portions.shape == self.path_list[i + 1].route_portions.shape)
    tmp_sum = np.zeros(self.path_list[0].route_portions.shape)
    for i in range(len(self.path_list)):
      self.path_list[i].route_portions = np.maximum(self.path_list[i].route_portions, 1e-6)
      tmp_sum += self.path_list[i].route_portions
    for i in range(len(self.path_list)):
      self.path_list[i].route_portions = self.path_list[i].route_portions / tmp_sum
    if sum_to_OD:
      return tmp_sum

  def __str__(self):
    return "MNM_pathset, O node: {}, D node: {}, number_of_paths: {}".format(self.origin_node, self.destination_node, len(self.path_list))

  def __repr__(self):
    return self.__str__()

class MNM_pathtable():
  """
  A class to represent a path table for a transportation network.

  Attributes
  ----------
  path_dict : dict
    A dictionary to store path sets, indexed by origin and destination nodes.
  ID2path : OrderedDict
    An ordered dictionary to store paths indexed by their IDs.

  Methods
  -------
  add_pathset(pathset, overwriting=False):
    Adds a path set to the path table.
  build_from_file(file_name, w_ID=False):
    Builds the path table from a file.
  load_route_choice_from_file(file_name, w_ID=False):
    Loads route choice portions from a file and attaches them to paths.
  __str__():
    Returns a string representation of the path table.
  __repr__():
    Returns a string representation of the path table.
  generate_table_text():
    Generates a text representation of the path table.
  generate_portion_text():
    Generates a text representation of the route choice portions.
  """
  def __init__(self):
    print("MNM_pathtable")
    self.path_dict = dict()
    self.ID2path = OrderedDict()

  def add_pathset(self, pathset, overwriting = False):
    if pathset.origin_node not in self.path_dict.keys():
      self.path_dict[pathset.origin_node] = dict()
    if (not overwriting) and (pathset.destination_node in 
              self.path_dict[pathset.origin_node]):
      raise ("Error: exists pathset in the pathtable")
    else:
      self.path_dict[pathset.origin_node][pathset.destination_node] = pathset

  def build_from_file(self, file_name, w_ID = False):
    if w_ID:
      raise ("Error, path table build_from_file no implemented")
    self.path_dict = dict()
    self.ID2path = OrderedDict()
    # f = file(file_name)
    f = open(file_name, "r")
    log = f.readlines()
    f.close()
    for i in range(len(log)):
      tmp_str = log[i]
      if tmp_str == '':
        continue
      words = tmp_str.split()
      origin_node = int(words[0])
      destination_node = int(words[-1])
      if origin_node not in self.path_dict.keys():
        self.path_dict[origin_node] = dict()
      if destination_node not in self.path_dict[origin_node].keys():
        tmp_path_set = MNM_pathset()
        tmp_path_set.origin_node = origin_node
        tmp_path_set.destination_node = destination_node
        self.add_pathset(tmp_path_set)
      tmp_node_list = list(map(lambda x : int(x), words))
      tmp_path = MNM_path(tmp_node_list, i)
      self.path_dict[origin_node][destination_node].add_path(tmp_path)
      self.ID2path[i] = tmp_path

  def load_route_choice_from_file(self, file_name, w_ID = False):
    if w_ID:
      raise ("Error, pathtable load_route_choice_from_file not implemented")
    # f = file(file_name)
    f = open(file_name, "r")
    log = list(filter(lambda x: not x.strip() == '', f.readlines()))
    f.close()
    assert(len(log) == len(self.ID2path))
    for i in range(len(log)):
      tmp_portions = np.array(log[i].strip().split()).astype(float)
      self.ID2path[i].attach_route_choice_portions(tmp_portions)

  def __str__(self):
    return "MNM_pathtable, number of paths:" + str(len(self.ID2path)) 

  def __repr__(self):
    return self.__str__()

  def generate_table_text(self):
    tmp_str = ""
    # python 2
    # for path_ID, path in self.ID2path.iteritems():
    # python 3
    for path_ID, path in self.ID2path.items():
      tmp_str += path.generate_node_list_text() + '\n'
    return tmp_str

  def generate_portion_text(self):
    tmp_str = ""
    # python 2
    # for path_ID, path in self.ID2path.iteritems():
    # python 3
    for path_ID, path in self.ID2path.items():
      tmp_str += path.generate_portion_text() + '\n'
    return tmp_str    

# class MNM_routing():
#   def __init__(self):
#     print "MNM_routing"

# class MNM_routing_fixed(MNM_routing):
#   def __init__(self):
#     super(MNM_routing_fixed, self).__init__()
#     self.required_items = ['num_path', 'choice_portion', 'route_frq', 'path_table']

# class MNM_routing_adaptive(MNM_routing):
#   """docstring for MNM_routing_adaptive"""
#   def __init__(self):
#     super(MNM_routing_adaptive, self).__init__()
#     self.required_items = ['route_frq']

# class MNM_routing_hybrid(MNM_routing):
#   """docstring for MNM_routing_hybrid"""
#   def __init__(self):
#     super(MNM_routing_hybrid, self).__init__()
#     self.required_items = []
    
    

class MNM_config():
  """
  A class to represent the configuration for MNM (Macroscopic Network Model).

  Attributes
  ----------
  config_dict : OrderedDict
    A dictionary to store configuration parameters.
  type_dict : dict
    A dictionary to map configuration parameter names to their types.

  Methods
  -------
  __init__():
    Initializes the MNM_config object with default values.
  build_from_file(file_name):
    Builds the configuration dictionary from a given file.
  __str__():
    Returns a string representation of the configuration.
  __repr__():
    Returns a string representation of the configuration (same as __str__()).
  """
  def __init__(self):
    print("MNM_config")
    self.config_dict = OrderedDict()
    self.type_dict = {
      # DTA
      'network_name': str, 
      'unit_time': int, 
      'total_interval': int,
      'assign_frq' : int, 
      'start_assign_interval': int, 
      'max_interval': int,
      'flow_scalar': int, 
      'num_of_link': int, 
      'num_of_node': int, 
      'num_of_O': int, 
      'num_of_D': int, 
      'OD_pair': int,
      'multi_OD_seq': int,
      'routing_type' : str,
      'adaptive_ratio': float,
      'init_demand_split': int, 
      'num_of_tolled_link': int, 
      'num_of_vehicle_labels': int,
      'ev_label': int,
      'num_of_charging_station': int,

      'EV_starting_range_roadside_charging': float,
      'EV_starting_range_non_roadside_charging': float,
      'EV_full_range': float,

      # STAT
      'rec_mode': str, 
      'rec_mode_para': str, 
      'rec_folder': str,
      'rec_volume': int, 
      'volume_load_automatic_rec': int, 
      'volume_record_automatic_rec': int,
      'rec_tt': int, 
      'tt_load_automatic_rec': int, 
      'tt_record_automatic_rec': int, 
      'rec_gridlock': int,

      # FIXED, ADAPTIVE
      'route_frq': int, 
      'vot': float, 
      'path_file_name': str, 
      'num_path': int,
      'choice_portion': str,
      'buffer_length':int
    }

  def build_from_file(self, file_name):
    self.config_dict = OrderedDict()
    # f = file(file_name)
    f = open(file_name, "r")
    log = f.readlines()
    f.close()
    for i in range(len(log)):
      tmp_str = log[i]
      # print tmp_str
      if tmp_str == '' or tmp_str.startswith('#') or tmp_str.strip() == '':
        continue
      if tmp_str.startswith('['):
        new_item = tmp_str.strip().strip('[]')
        if new_item in self.config_dict.keys():
          raise ("Error, MNM_config, multiple items", new_item)
        self.config_dict[new_item] = OrderedDict()
        continue
      words = tmp_str.split('=')
      name = words[0].strip()
      self.config_dict[new_item][name] = self.type_dict[name](words[1].strip())

  def __str__(self):
    tmp_str = ''

    tmp_str += '[DTA]\n'
    # python 2
    # for name, value in self.config_dict['DTA'].iteritems():
    # python 3
    for name, value in self.config_dict['DTA'].items():
      tmp_str += "{} = {}\n".format(str(name), str(value))

    tmp_str += '\n[STAT]\n'
    # python 2
    # for name, value in self.config_dict['STAT'].iteritems():
    # python 3
    for name, value in self.config_dict['STAT'].items():
      tmp_str += "{} = {}\n".format(str(name), str(value))  

    if self.config_dict['DTA']['routing_type'] in ['Fixed', 'Adaptive', 'Hybrid']:
      tmp_str += '\n[FIXED]\n'
      # python 2
      # for name, value in self.config_dict['FIXED'].iteritems():
      # python 3
      for name, value in self.config_dict['FIXED'].items():
        tmp_str += "{} = {}\n".format(str(name), str(value))  

    if self.config_dict['DTA']['routing_type'] in ['Fixed', 'Adaptive', 'Hybrid']:
      tmp_str += '\n[ADAPTIVE]\n'
      # python 2
      # for name, value in self.config_dict['ADAPTIVE'].iteritems():
      # python 3
      for name, value in self.config_dict['ADAPTIVE'].items():
        tmp_str += "{} = {}\n".format(str(name), str(value))

    return tmp_str

  def __repr__(self):
    return self.__str__()

class MNM_network_builder():
  """
  A class used to build and manage a transportation network.

  Attributes
  ----------
  config : MNM_config
    Configuration settings for the network.
  network_name : str or None
    The name of the network.
  link_list : list
    A list of links in the network.
  node_list : list
    A list of nodes in the network.
  graph : MNM_graph
    The graph representation of the network.
  od : MNM_od
    The origin-destination pairs in the network.
  demand : MNM_demand
    The demand data for the network.
  path_table : MNM_pathtable
    The table of paths in the network.
  route_choice_flag : bool
    A flag indicating if route choice portions are loaded.

  Methods
  -------
  get_link(ID)
    Returns the link with the specified ID.
  load_from_folder(path, config_file_name='config.conf', link_file_name='MNM_input_link', node_file_name='MNM_input_node', graph_file_name='Snap_graph', od_file_name='MNM_input_od', pathtable_file_name='path_table', path_p_file_name='path_table_buffer', demand_file_name='MNM_input_demand')
    Loads network data from the specified folder.
  dump_to_folder(path, config_file_name='config.conf', link_file_name='MNM_input_link', node_file_name='MNM_input_node', graph_file_name='Snap_graph', od_file_name='MNM_input_od', pathtable_file_name='path_table', path_p_file_name='path_table_buffer', demand_file_name='MNM_input_demand')
    Dumps network data to the specified folder.
  read_link_input(file_name)
    Reads link data from the specified file.
  read_node_input(file_name)
    Reads node data from the specified file.
  generate_link_text()
    Generates text representation of the link data.
  generate_node_text()
    Generates text representation of the node data.
  set_network_name(name)
    Sets the name of the network.
  update_demand_path(f)
    Updates the demand path with the specified data.
  get_path_flow()
    Returns the path flow data.
  get_route_portion_matrix()
    Returns the route portion matrix.
  """
  def __init__(self):
    self.config = MNM_config()
    self.network_name = None
    self.link_list = list()
    self.node_list = list()
    self.graph = MNM_graph()
    self.od = MNM_od()
    self.demand = MNM_demand()
    self.path_table = MNM_pathtable()
    self.route_choice_flag = False


  def get_link(self, ID):
    """
    Retrieve a link from the link list by its ID.

    Args:
      ID (int): The unique identifier of the link to retrieve.

    Returns:
      Link: The link object with the specified ID if found, otherwise None.
    """
    for link in self.link_list:
      if link.ID == ID:
        return link
    return None

  def load_from_folder(self, path, config_file_name = 'config.conf',
                                    link_file_name = 'MNM_input_link', node_file_name = 'MNM_input_node',
                                    graph_file_name = 'Snap_graph', od_file_name = 'MNM_input_od',
                                    pathtable_file_name = 'path_table', path_p_file_name = 'path_table_buffer',
                                    demand_file_name = 'MNM_input_demand'):
    if os.path.isfile(os.path.join(path, config_file_name)):
      self.config.build_from_file(os.path.join(path, config_file_name))
    else:
      print("No config file")
    if os.path.isfile(os.path.join(path, link_file_name)):
      self.link_list = self.read_link_input(os.path.join(path, link_file_name))
    else:
      print("No link input")
    if os.path.isfile(os.path.join(path, node_file_name)):
      self.node_list = self.read_node_input(os.path.join(path, node_file_name))
    else:
      print("No node input")
    if os.path.isfile(os.path.join(path, graph_file_name)):
      self.graph.build_from_file(os.path.join(path, graph_file_name))
    else:
      print("No graph input")

    if os.path.isfile(os.path.join(path, od_file_name)):
      self.od.build_from_file(os.path.join(path, od_file_name))
    else:
      print("No OD input")

    if os.path.isfile(os.path.join(path, demand_file_name)):
      self.demand.build_from_file(os.path.join(path, demand_file_name))
    else:
      print("No demand input")

    if os.path.isfile(os.path.join(path, pathtable_file_name)):
      self.path_table.build_from_file(os.path.join(path, pathtable_file_name))
      if os.path.isfile(os.path.join(path, path_p_file_name)):
        self.path_table.load_route_choice_from_file(os.path.join(path, path_p_file_name))
        self.route_choice_flag = True
      else:
        self.route_choice_flag = False
        print("No route choice portition for path table")
    else:
      print("No path table input")


  def dump_to_folder(self, path, config_file_name = 'config.conf',
                                    link_file_name = 'MNM_input_link', node_file_name = 'MNM_input_node',
                                    graph_file_name = 'Snap_graph', od_file_name = 'MNM_input_od',
                                    pathtable_file_name = 'path_table', path_p_file_name = 'path_table_buffer',
                                    demand_file_name = 'MNM_input_demand'):
    if not os.path.isdir(path):
      os.makedirs(path)

    # python 2
    # f = open(os.path.join(path, link_file_name), 'wb')
    # python 3
    f = open(os.path.join(path, link_file_name), 'w')
    f.write(self.generate_link_text())
    f.close()

    # python 2
    # f = open(os.path.join(path, node_file_name), 'wb')
    # python 3
    f = open(os.path.join(path, node_file_name), 'w')
    f.write(self.generate_node_text())
    f.close()

    # python 2
    # f = open(os.path.join(path, config_file_name), 'wb')
    # python 3
    f = open(os.path.join(path, config_file_name), 'w')
    f.write(str(self.config))
    f.close()

    # python 2
    # f = open(os.path.join(path, od_file_name), 'wb')
    # python 3
    f = open(os.path.join(path, od_file_name), 'w')
    f.write(self.od.generate_text())
    f.close()

    # python 2
    # f = open(os.path.join(path, demand_file_name), 'wb')
    # python 3
    f = open(os.path.join(path, demand_file_name), 'w')
    f.write(self.demand.generate_text())
    f.close()

    # python 2
    # f = open(os.path.join(path, graph_file_name), 'wb')
    # python 3
    f = open(os.path.join(path, graph_file_name), 'w')
    f.write(self.graph.generate_text())
    f.close()

    # python 2
    # f = open(os.path.join(path, pathtable_file_name), 'wb')
    # python 3
    f = open(os.path.join(path, pathtable_file_name), 'w')
    f.write(self.path_table.generate_table_text())
    f.close()

    if self.route_choice_flag:
      # python 2
      # f = open(os.path.join(path, path_p_file_name), 'wb')
      # python 3
      f = open(os.path.join(path, path_p_file_name), 'w')
      f.write(self.path_table.generate_portion_text())
      f.close()

  def read_link_input(self, file_name):
    link_list = list()
    # f = file(file_name)
    f = open(file_name, "r")
    log = f.readlines()[1:]
    f.close()
    for i in range(len(log)):
      tmp_str = log[i].strip()
      if tmp_str == '':
        continue
      words = tmp_str.split()
      ID = int(words[0])
      typ = words[1]
      length = float(words[2])
      ffs = float(words[3])
      cap = float(words[4])
      rhoj = float(words[5])
      lanes = int(words[6])
      l = MNM_dlink(ID, length, typ, ffs, cap, rhoj, lanes)
      l.is_ok()
      link_list.append(l)
    return link_list

  def read_node_input(self, file_name):
    node_list = list()
    f = open(file_name, "r")
    log = f.readlines()[1:]
    f.close()
    for i in range(len(log)):
      tmp_str = log[i].strip()
      if tmp_str == '':
        continue
      words = tmp_str.split()
      ID = int(words[0])
      typ = words[1]
      n = MNM_dnode(ID, typ)
      n.is_ok()
      node_list.append(n)
    return node_list

  def generate_link_text(self):
    tmp_str = '#ID Type LEN(mile) FFS(mile/h) Cap(v/hour) RHOJ(v/miles) Lane\n'
    for link in self.link_list:
      tmp_str += link.generate_text() + '\n'
    return tmp_str

  def generate_node_text(self):
    tmp_str = '#ID Type\n'
    for node in self.node_list:
      tmp_str += node.generate_text() + '\n'
    return tmp_str

  def set_network_name(self, name):
    self.network_name = name

  def update_demand_path(self, f):
    """
    Updates the demand path based on the provided demand matrix.
    This function reshapes the input demand matrix `f` to match the number of 
    paths and the maximum interval defined in the configuration. It then updates 
    the route choice portions for each path and normalizes the route portions 
    for each origin-destination (OD) pair. Finally, it updates the demand dictionary 
    and sets the route choice flag to True.
    Args:
      f (numpy.ndarray): A 1D array representing the demand matrix. The length 
                 of the array should be equal to the product of the 
                 number of paths and the maximum interval.
    Raises:
      AssertionError: If the length of `f` does not match the expected size 
              based on the number of paths and the maximum interval.
    """
    assert (len(f) == len(self.path_table.ID2path) * self.config.config_dict['DTA']['max_interval'])
    f = f.reshape(self.config.config_dict['DTA']['max_interval'], len(self.path_table.ID2path))
    for i, path_ID in enumerate(self.path_table.ID2path.keys()):
      path = self.path_table.ID2path[path_ID]
      path.attach_route_choice_portions(f[:, i])
    self.demand.demand_dict = dict()
    for O_node in self.path_table.path_dict.keys():
      for D_node in self.path_table.path_dict[O_node].keys():
        demand = self.path_table.path_dict[O_node][D_node].normalize_route_portions(sum_to_OD = True)
        self.demand.add_demand(self.od.O_dict.inv[O_node], self.od.D_dict.inv[D_node], demand, overwriting = True)
    self.route_choice_flag = True

  def get_path_flow(self):
    """
    Calculate the path flow matrix for the network.
    This method computes the path flow matrix `f` where each element `f[t, i]` 
    represents the flow on path `i` at time interval `t`. The flow is calculated 
    based on the route portions of each path and the demand between origin-destination pairs.
    Returns:
      numpy.ndarray: A flattened array representing the path flow matrix.
    """
    f = np.zeros((self.config.config_dict['DTA']['max_interval'], len(self.path_table.ID2path)))
    for i, ID in enumerate(self.path_table.ID2path.keys()):
      path = self.path_table.ID2path[ID]
      # print path.route_portions
      f[:, i] = path.route_portions * self.demand.demand_dict[self.od.O_dict.inv[path.origin_node]][self.od.D_dict.inv[path.destination_node]]
    return f.flatten()

  def get_route_portion_matrix(self):
    """
    Generates a route portion matrix for the transportation network.
    This method constructs a sparse matrix where each element represents the 
    portion of a specific path used for a given origin-destination (OD) pair 
    over different time intervals. The matrix is constructed using the 
    Compressed Sparse Row (CSR) format for efficient storage and computation.
    Returns:
      scipy.sparse.csr_matrix: A sparse matrix of shape 
      (num_one_path * num_intervals, num_one_OD * num_intervals) where 
      num_one_path is the number of unique paths, num_intervals is the 
      number of time intervals, and num_one_OD is the number of OD pairs.
    """
    num_intervals = self.config.config_dict['DTA']['max_interval']
    for O_node in self.path_table.path_dict.keys():
      for D_node in self.path_table.path_dict[O_node].keys():
        self.path_table.path_dict[O_node][D_node].normalize_route_portions(sum_to_OD = False)
    path_ID2idx = {ID:idx for idx, ID in enumerate(self.path_table.ID2path.keys())}
    col = list()
    row = list()
    val = list()
    num_one_OD = len(self.demand.demand_list)
    num_one_path = len(self.path_table.ID2path)

    for OD_idx, (O,D) in enumerate(self.demand.demand_list):
      tmp_path_set = self.path_table.path_dict[self.od.O_dict[O]][self.od.D_dict[D]]
      for tmp_path in tmp_path_set.path_list:
        path_idx = path_ID2idx[tmp_path.path_ID]
        for ti in range(num_intervals):
          row.append(path_idx + ti * num_one_path)
          col.append(OD_idx + ti * num_one_OD)
          val.append(tmp_path.route_portions[ti])

    P = coo_matrix((val, (row, col)), 
                   shape=(num_one_path * num_intervals, num_one_OD * num_intervals)).tocsr()
    return P
  
  def plot_network_by_nx(self, ax):
    """
    Plots the network using NetworkX.
    This function visualizes the network graph using NetworkX's drawing capabilities.
    It plots nodes, edges, and labels on the provided Matplotlib axis.
    Parameters:
    ax (matplotlib.axes.Axes): The Matplotlib axis on which to plot the network.
    Returns:
    matplotlib.axes.Axes: The axis with the plotted network.
    """
    
    # Now let's plot the graph
    g = self.graph.G # get networkx directed graph
    # Create a layout for our nodes 
    layout = nx.spring_layout(g)

    # Extract node types and map to colors
    node_id_to_type = {node.ID: node.typ for node in self.node_list}
    node_colors = [DNODE_COLOR_CODE[node_id_to_type[node_id]] for node_id in g.nodes()]

    # Draw the nodes
    nx.draw_networkx_nodes(g, layout, node_size=500, node_color=node_colors, ax=ax)

    # Draw the edges
    nx.draw_networkx_edges(g, layout, arrows=True, edge_color="lightgrey", ax=ax)

    # Add labels to the nodes
    nx.draw_networkx_labels(g, layout, font_size=12, font_family='sans-serif', ax=ax)

    # Add edge labels (using the edge_id)
    edge_labels = {(edge[0], edge[1]): edge[2]["ID"] for edge in g.edges(data=True)}
    nx.draw_networkx_edge_labels(g, layout, edge_labels=edge_labels, ax=ax)

    # Remove axis
    ax.axis('off')
    return ax
  
  def plot_network_by_pyvis(self, data=None, edge_color_column=None, notebook=True, output_html="network.html"):
    """
    Visualizes the network using Pyvis and NetworkX.
    Parameters:
    -----------
    data : dict, optional
      A dictionary containing edge data. The keys are edge IDs and the values are dictionaries with edge attributes to be visualized.
    edge_color_column : str, optional
      The key in the `data` dictionary whose values will be used to color the edges.
    notebook : bool, default=True
      If True, the visualization will be displayed in a Jupyter notebook.
    output_html : str, default="network.html"
      The name of the output HTML file where the network visualization will be saved.
    Returns:
    --------
    None
    """
    from pyvis.network import Network
    # Get the NetworkX directed graph
    g = self.graph.G

    # Create a Pyvis Network
    net = Network(notebook=True, directed=True)

    # Extract node types and map to colors
    node_id_to_type = {node.ID: node.typ for node in self.node_list}
    for node_id in g.nodes():
        net.add_node(node_id, color=DNODE_COLOR_CODE[node_id_to_type[node_id]], label=str(node_id), labelPosition="top")

    # Extract edge types and map to colors and styles
    edge_id_to_stats = {edge.ID: {"type": edge.typ,
                                "length": edge.length*50.,
                                "ffs": edge.ffs/20. if edge.typ =="FWY" else 5,
                                } for edge in self.link_list}
    
    if edge_color_column is not None and data is not None:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Normalize the color value to be between 0 and 1
        norm = mcolors.Normalize(vmin=min(data[edge_id][edge_color_column] for edge_id in data), 
                        vmax=max(data[edge_id][edge_color_column] for edge_id in data))
        cmap = plt.get_cmap('Wistia')  # You can choose any colormap you like
    
    for edge in g.edges(data=True):
        edge_id = edge[2]["ID"]
        title_text = f"ID:{edge_id}\nLength: {edge_id_to_stats[edge_id]['length']}" if data is None else f"ID:{edge_id}\n"+'\n'.join([f"{k}: {v}" for k, v in data[edge_id].items()])
        color = "lightgrey" 
        if edge_color_column is not None and data is not None:
            color_val = data[edge_id][edge_color_column]
            # Get the color from the colormap
            color = mcolors.to_hex(cmap(norm(color_val)))
        net.add_edge(edge[0], edge[1], color=color,
                    value=edge_id_to_stats[edge_id]["ffs"], 
                    length=edge_id_to_stats[edge_id]["length"], 
                    label=str(edge_id), 
                    dashes=DLINK_FORMAT_CODE[edge_id_to_stats[edge_id]["type"]],
                    title=title_text, 
                    labelHighlightBold=True, 
                    labelPosition="middle")
        
    # Generate the network visualization
    net.show(output_html)