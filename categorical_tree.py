"""
This module contains a categorical decision tree

Example
--------
import sklearn
import categorical_tree
import numpy

n_classes=1 
n_features=6
capacity = 10
max_children = 5

feature_names = numpy.array(["0thFeature","1stFeature","2ndFeature","3rdFeature","4thFeature","5thFeature"], dtype=("a40"), order='F')
is_categorical = numpy.array([False,False,False,False,True,False], dtype=("b"), order='F')

mytree=categorical_tree.Categorical_Tree(n_classes,n_features, feature_names, is_categorical, capacity, max_children)

mytree._add_split_node(-1,0,3,[0.5,100000],0,0,250,700) # the root, node 0
mytree._add_split_node(0 ,0,5,[65.3,77.5,100000],0,0,50,200) # node 1
mytree._add_leaf(0,1,800,0,200) # node 2
mytree._add_leaf(1,0,100,0,20) # node 3
mytree._add_leaf(1,1,200,0,20) # node 4
mytree._add_split_node(1,2,4,[["str1","str2"],["str3","str4"]],0,0,10,300) # node 5
mytree._add_leaf(5,0,250,0,5) # node 6
mytree._add_leaf(5,1,350,0,5) # node 7

X=numpy.array([[0,0,0,0.5,"str1",70.0]], dtype='object', order='F')
mytree.predict(X[0,:],0)

out_file = categorical_tree.export_graphviz(mytree, out_file="/home/tstrinopoulos/src/scikit-learn-0.11/sklearn/tree/mytree.dot")
out_file.close()

mytree2=categorical_tree.Categorical_Tree(n_classes,n_features, feature_names, is_categorical, capacity, max_children)
categorical_tree.import_graphviz(mytree2, "mytree.dot")

out_file = categorical_tree.export_graphviz(mytree2, out_file="/home/tstrinopoulos/src/scikit-learn-0.11/sklearn/tree/mytree2.dot")
out_file.close()

"""

# Code was adapted from scikit.
# Code is originally adapted from MILK: Machine Learning Toolkit
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# License: MIT. See COPYING.MIT file in the milk distribution

# Authors: Brian Holt, Peter Prettenhofer, Satrajit Ghosh, Gilles Louppe
# License: BSD3


from __future__ import division
import numpy as np

from sklearn.tree import _tree  # this in not needed since I will write my own predict function

__all__ = ["Categorical_Tree",  # expose the Categorical_Tree class
           "export_graphviz",
           "import_graphviz"]   # expose graphviz



def export_graphviz(decision_tree, out_file = None, color = None):
    """Export a decision tree in DOT format.

    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)
    The output can be viewed with 

        $ gv tree.ps

    The \n character cannot be viewed by bdotty so I am outputting here \N instead.
    To export to ps I can use 
        $ more mytree.dot | sed 's/\\N/\\n/g' | dot -Tps -o mytree.ps
        $ ./bdotty mytree.dot &

    Turn off the fixed size:
        $ more mytree.dot | sed 's/\\N/\\n/g' | sed 's/fixedsize=true/fixedsize=false/g' | dot -Tps -o mytree.ps

    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to graphviz.

    out : file object or string, optional (default=None)
        Handle or name of the output file.


    Returns
    -------
    out_file : file object
        The file object to which the tree was exported.  The user is
        expected to `close()` this object when done with it.
    
    """
    def node_to_str(tree, node_id, color = None):
        # if the node is a leaf
        if tree.feature[node_id] < 0:
            output_string = "node_id = %s\\Nerror = %.4f\\Nsamples = %s\\Nvalue = %.0f" \
                   % (node_id,tree.init_error[node_id], tree.n_samples[node_id],
                      tree.value[node_id])
        else:
            # if the node is a categorical split
            if tree.is_categorical[tree.feature[node_id]]:
                output_string = "node_id = %s\\N%s in " \
                %(node_id, tree.feature_names[tree.feature[node_id]])
                for idx in range(tree.children_number[node_id]):
                    output_string += "%s\\N" \
                    %tree.threshold[node_id, idx]
            # if the node is an ordinal split
            else:
                output_string = "node_id = %s\\N%s : " \
                %(node_id, tree.feature_names[tree.feature[node_id]])
                for idx in range(tree.children_number[node_id]):
                    output_string += "<%s\\N" \
                    %tree.threshold[node_id, idx]

            output_string += "error = %s\\Nsamples = %s\\Nvalue = %.0f" \
                % (tree.init_error[node_id], 
                  tree.n_samples[node_id],tree.value[node_id])
            if color != None:
                output_string += '", color= "'+color[tree.feature[node_id]]

        return output_string.replace("[","").replace("]","")                 

    recdepth = 0
    # traverses the tree and saves the nodes to the outfile
    def recurse(tree, node_id, recdepth, parent=None, maxdepth = 50, color = None):
        recdepth +=1
        if recdepth > maxdepth:
            return -999
        if node_id == Categorical_Tree.LEAF:
            raise ValueError("Invalid node_id %s" % Categorical_Tree.LEAF)
        #left_child, right_child = tree.children[node_id, :]

        # add node with description
        out_file.write('"%d" [label="%s", shape="box"] ;\n' %
                (node_id, node_to_str(tree, node_id, color)))

        if not parent is None:
            # add edge to parent
            out_file.write('"%d" -> "%d" ;\n' % (parent, node_id))

        if not (tree.feature[node_id] == Categorical_Tree.LEAF):
            for idx in range(tree.children_number[node_id]):
                recurse(tree, tree.children[node_id, idx], recdepth, node_id, maxdepth, color)

    if out_file is None:
        out_file = open("tree.dot", "w")
    elif isinstance(out_file, basestring):
        out_file = open(out_file, "w")

    out_file.write("digraph Categorical_Tree {\n")
    out_file.write("graph [ bgcolor=lightgray, resolution=128, fontname=Arial, fontcolor=blue, fontsize=12 ];\n")
    out_file.write("node [ fixedsize=true, width=1, fontname=Arial, fontcolor=blue, fontsize=11];\n")
    out_file.write("edge [ fontname=Helvetica, fontcolor=red, fontsize=10 ];\n")

    recurse(decision_tree, 0, recdepth, None, 50, color)

    # add rank to .dot file
    unique_features = []
    for feature in decision_tree.feature:
        if feature not in unique_features and feature >= 0:
            unique_features.append(feature)
    for feature in unique_features:
        out_file.write("{rank = same; ")
        for idx in range(decision_tree.node_count):
            if decision_tree.feature[idx] == feature:
                out_file.write("%d; " % (idx))
        out_file.write("};\n")
            
    out_file.write("}")

    return out_file

def import_graphviz(tree, in_file_name= None):
    """Import a decision tree from a file in DOT format.
    The in_file_name must be in a format slightly more general than what export_graphviz outputs.
    It can be editted in Windows, but not with dotty, because dotty renames nodes.

    Special characters that are not allowed in the categorical fields:","
    """
    if in_file_name is None:
        in_file_ptr = open("tree.dot")
    elif isinstance(in_file_name, basestring):
        in_file_ptr= open(in_file_name)

    in_file = in_file_ptr.read()
    # edges
    for line in in_file.replace("\t","").replace("\n","").replace(";","").split("]"):       
        if "->" in line:
            parent_id = int(line.replace("\n","").split("->")[0].replace(" ","").replace('"',""))
            child_id = int(line.replace("\n","").split("->")[1].lstrip().split(" ")[0].replace('"',""))
            if tree.children_number[parent_id] == Categorical_Tree.UNDEFINED:
                tree.children_number[parent_id] = 1
            else:
                tree.children_number[parent_id] += 1
            tree.children[parent_id,tree.children_number[parent_id]-1] = child_id

    # nodes
    for line in in_file.replace("\t","").replace("\n","").split("]"):        
        if "label" in line:
            tree.node_count +=1
            # get the contents of the label tag
            line = line.split("label")[1].split('"')[1]
            # leaf
            if len(line.split("\N"))==4:
                node_id = int(line.split("\N")[0].replace(" ","").split("=")[1])
                tree.init_error[node_id] = line.split("\N")[1].replace(" ","").split("=")[1]
                tree.n_samples[node_id] = line.split("\N")[2].replace(" ","").split("=")[1]
                tree.value[node_id] = line.split("\N")[3].replace(" ","").split("=")[1]
            # split node
            else:
                node_id = int(line.split("\N")[0].replace(" ","").split("=")[1])
                # find the feature we are splitting on
                feature_name_to_split=line.split("\N")[1].split(" ")[0]
                found = False
                for idx,fname in enumerate(tree.feature_names):
                    if(feature_name_to_split == fname):
                        tree.feature[node_id] = idx
                        found = True
                if not found:
                    print "feature to split on not found. check feature names"

                for idx in range(tree.children_number[node_id]):
                    if tree.is_categorical[tree.feature[node_id]]:
                        line2 = line.split("\N")[idx+1]
                        if "in" in line2:
                            line2 = line2.split("in ")[1] 
                        tree.threshold[node_id, idx]=line2.replace(" ","").replace("'","").split(",")
                    else:
                        tree.threshold[node_id, idx]=float(line.split("\N")[idx+1].split("<")[1])

                tree.init_error[node_id] = line.split("\N")[tree.children_number[node_id]+1].replace(" ","").split("=")[1]
                tree.n_samples[node_id] = line.split("\N")[tree.children_number[node_id]+2].replace(" ","").split("=")[1]
                tree.value[node_id] = line.split("\N")[tree.children_number[node_id]+3].replace(" ","").split("=")[1]





class Categorical_Tree(object):
    """Struct-of-arrays representation of a binary decision tree that allows for 
    categorical variables

    The binary tree is represented as a number of parallel arrays.
    The i-th element of each array holds information about the
    node `i`. You can find a detailed description of all arrays
    below. NOTE: Some of the arrays only apply to either leaves or
    split nodes, resp. In this case the values of nodes of the other
    type are arbitrary!

    Attributes
    ----------
    node_count : int
        Number of nodes (internal nodes + leaves) in the tree.

    children : np.ndarray, shape=(node_count, 2), dtype=int32
        `children[i, 0]` holds the node id of the left child of node `i`.
        `children[i, 1]` holds the node id of the right child of node `i`.
        For leaves `children[i, 0] == children[i, 1] == Categorical_Tree.LEAF == -1`.

    feature : np.ndarray of int32
        The feature to split on (only for internal nodes).

    feature_names : np.ndarray of string
        The names of each variable

    is_categorical : np.ndarray of bool
        Whether the variable is categorical or not

    threshold : np.ndarray of float64
        The threshold to branch to each child of every node (only for split_nodes).

    value : np.ndarray of float64, shape=(capacity, n_classes)
        Contains the constant prediction value of each node.

    best_error : np.ndarray of float64
        The error of the (best) split.
        For leaves `init_error == `best_error`.

    init_error : np.ndarray of float64
        The initial error of the node (before splitting).
        For leaves `init_error == `best_error`.

    n_samples : np.ndarray of np.int32
        The number of samples at each node.

    n_classes : number of classes to classify into

    parent :  np.ndarray, shape=(node_count, 2), dtype=int32
        The parent of the node.   
    """

    LEAF = -1
    UNDEFINED = -2

    def __init__(self, n_classes, n_features, feature_names = None, 
                is_categorical = None, capacity = 3, max_children = 2):
        #empty does not set values to zero, but to random values
        self.n_classes = n_classes 
        self.n_features = n_features

        self.node_count = 0

        self.children = np.empty((capacity, max_children), dtype=np.int32)
        self.children.fill(Categorical_Tree.UNDEFINED)

        self.children_number = np.empty((capacity,), dtype=np.int32)
        self.children_number.fill(Categorical_Tree.UNDEFINED)

        self.parent = np.empty((capacity,), dtype=np.int32)
        self.parent.fill(Categorical_Tree.UNDEFINED)

        self.feature = np.empty((capacity,), dtype=np.int32)
        self.feature.fill(Categorical_Tree.UNDEFINED)

        #features_names can have at most 40 characters
        self.feature_names = np.empty((n_features,), dtype=("a40"))
        if feature_names !=None:
            self.feature_names=feature_names
        else:
            self.feature_names.fill("Undefined")

        self.is_categorical = np.empty((n_features,), dtype=("b"))
        if feature_names != None:
            self.is_categorical=is_categorical

        self.threshold = np.empty((capacity, max_children), dtype='object')
        self.value = np.empty((capacity, n_classes), dtype=np.float64)        

        self.best_error = np.empty((capacity,), dtype=np.float32)
        self.init_error = np.empty((capacity,), dtype=np.float32)
        self.n_samples = np.empty((capacity,), dtype=np.int32)

    def _resize(self, capacity=None):
        """Resize tree arrays to `capacity` of nodes, if `None` double capacity. """
        if capacity is None:
            capacity = int(self.children.shape[0] * 2.0)

        if capacity == self.children.shape[0]:
            return

        self.children.resize((capacity, 2), refcheck=False)
        self.parent.resize((capacity, 2), refcheck=False)
        self.feature.resize((capacity,), refcheck=False)
        self.threshold.resize((capacity,), refcheck=False)
        self.value.resize((capacity, self.value.shape[1]), refcheck=False)
        self.best_error.resize((capacity,), refcheck=False)
        self.init_error.resize((capacity,), refcheck=False)
        self.n_samples.resize((capacity,), refcheck=False)

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

    def _add_split_node(self, parent, child_number, feature, threshold,
                        best_error, init_error, n_samples, value, node_id = None):
        """Add a splitting node to the tree. The new node registers itself as
        the child of its parent. 
        threshold is a list of split conditions from left to right in the graph of the tree
        """
        # allow for user-entered node_id
        if node_id == None:
            node_id = self.node_count

        if node_id >= self.children.shape[0]:
            self._resize()

        self.feature[node_id] = feature

        for idx, condition in enumerate(threshold):
            self.threshold[node_id, idx] = condition

        self.init_error[node_id] = init_error
        self.best_error[node_id] = best_error
        self.n_samples[node_id] = n_samples
        self.value[node_id] = value

        self.children_number[node_id] = len(threshold)

        # set as child of parent
        if parent > Categorical_Tree.LEAF:
            self.children[parent, child_number] = node_id

        self.parent[node_id]=parent

        self.node_count += 1
        return node_id

    def _add_leaf(self, parent,child_number, value, error, n_samples, node_id = None):
        """Add a leaf to the tree. The new node registers itself as the
        child of its parent. """
        # allow for user-entered node_id
        if node_id == None:
            node_id = self.node_count

        if node_id >= self.children.shape[0]: #gets the max count of the nodes
            self._resize()

        self.value[node_id] = value
        self.n_samples[node_id] = n_samples
        self.init_error[node_id] = error
        self.best_error[node_id] = error

        if parent > Categorical_Tree.LEAF:
            self.children[parent, child_number] = node_id

        self.children[node_id, :] = Categorical_Tree.LEAF

        self.parent[node_id]=parent

        self.node_count += 1
        return node_id

    def edit_node(self, node_id, value = None, error = None, n_samples = None):
        """Edit the values of a node of the tree. """
        if value != None:
            self.value[node_id] = value
        if n_samples != None:
            self.n_samples[node_id] = n_samples
        if error != None:
            self.init_error[node_id] = error
            self.best_error[node_id] = error

    def remove_node(self, node_id):
        """Remove a node of the tree. Probably should remove the other child 
        of the parent[node_id]"""
        # whether the parent had 1 or 2 children before deletion we enter LEAF in the child
        if self.children[self.parent[node_id], 0]==node_id:
            self.children[self.parent[node_id], 0] = Categorical_Tree.LEAF
        else:
            self.children[self.parent[node_id], 1] = Categorical_Tree.LEAF

        self.children[node_id,0] = Categorical_Tree.UNDEFINED
        self.children[node_id,1] = Categorical_Tree.UNDEFINED
        
        self.node_count -=1

        # change parent from split node to leaf
        # renumber nodes

    def predict(self, X, node_id, increase_count = False, recdepth = None, maxrecdepth = 50):
        """ Predict the value of a point according to the tree.
        If the split is on a categorical feature "others" at the end of the list of children is a valid choice.
        If increase_count is True, the routine increments the n_samples of each node visited during evaluation.
        When the point is not found in the tree, the nodes up to which it was found are still increased. 
        """
        if recdepth == None:
            recdepth = 1
        else:
            recdepth +=1
        if recdepth > maxrecdepth: 
            return -999
        # if you are at a split node
        if self.feature[node_id] != Categorical_Tree.UNDEFINED:
            found = False
            if self.is_categorical[self.feature[node_id]]:
                for idx in range(self.children_number[node_id]):
                    if X[self.feature[node_id]] in self.threshold[node_id][idx]:
                        if increase_count:
                            self.n_samples[node_id] += 1
                        found = True
                        return self.predict(X, self.children[node_id, idx], increase_count, recdepth, maxrecdepth)

                # enable 'others', only when at the last position
                if not found and self.threshold[node_id][self.children_number[node_id]-1][0] == 'others':
                        if increase_count:
                            self.n_samples[node_id] += 1
                        found = True     
                        return self.predict(X, self.children[node_id, self.children_number[node_id]-1], increase_count, recdepth, maxrecdepth)
                
            else:
                for idx in range(self.children_number[node_id]):
                    if X[self.feature[node_id]] <= self.threshold[node_id][idx]:
                        if increase_count:
                            self.n_samples[node_id] += 1
                        found = True
                        return self.predict(X, self.children[node_id, idx], increase_count, recdepth, maxrecdepth)
            if not found:
                print "cannot find value X: "+str(X)+" in the tree. Stopped at: ", node_id
        # if you are at a leaf
        else:
            if increase_count:
                self.n_samples[node_id] += 1
            return self.value[node_id]

    def calculate_interior_ev(self, node_id):
        """ Calculate the expected value of split nodes based on the values of the leaves.
        Assumes that all the leaves have a value in them.
        It recurses backwards from the leaves.
        Run check_n_samples first to make sure n_samples is a probability.
        If all samples of the children are zero it returns the simple average.  
        """
        exp_value = 0
        total_samples = 0
        simple_exp_value = 0
        temp = 0
        for idx in range(self.children_number[node_id]):
            # if the child is a split node then recurse
            if self.feature[self.children[node_id,idx]] != Categorical_Tree.UNDEFINED:
                temp = self.calculate_interior_ev(self.children[node_id,idx])
                exp_value += self.n_samples[self.children[node_id,idx]]*temp
                simple_exp_value += temp
                total_samples += self.n_samples[self.children[node_id,idx]]
            else:
                exp_value += self.n_samples[self.children[node_id,idx]]*self.value[self.children[node_id,idx]]
                simple_exp_value += self.value[self.children[node_id,idx]]
                total_samples += self.n_samples[self.children[node_id,idx]]
        if total_samples != 0:
            self.value[node_id] = exp_value/total_samples
            return self.value[node_id]  
        else:
            self.value[node_id] = simple_exp_value/self.children_number[node_id]
            return simple_exp_value/self.children_number[node_id]
      

    def check_n_samples(self, node_id):
        """
        Check that n_samples of the children add up to the n_samples of the parent
        so that n_samples is a probability
        """
        if self.feature[node_id] != Categorical_Tree. UNDEFINED:
            total_samples = 0
            for idx in range(self.children_number[node_id]):
                total_samples += self.n_samples[self.children[node_id,idx]]

            for idx in range(self.children_number[node_id]):
                self.check_n_samples(self.children[node_id,idx])

            if total_samples != self.n_samples[node_id]: 
                print "Sum of children samples is not parent sample at node: ", node_id



    def compute_feature_importances(self, method="gini"):
        """Computes the importance of each feature (aka variable).

        The following `method`s are supported:

          * "gini" : The difference of the initial error and the error of the
                     split times the number of samples that passed the node.
          * "squared" : The empirical improvement in squared error.

        Parameters
        ----------
        method : str, optional (default="gini")
            The method to estimate the importance of a feature. Either "gini"
            or "squared".
        """
        if method == "gini":
            method = lambda node: (self.n_samples[node] * \
                                     (self.init_error[node] -
                                      self.best_error[node]))
        elif method == "squared":
            method = lambda node: (self.init_error[node] - \
                                   self.best_error[node]) ** 2.0
        else:
            raise ValueError(
                'Invalid value for method. Allowed string '
                'values are "gini", or "mse".')

        importances = np.zeros((self.n_features,), dtype=np.float64)

        for node in range(self.node_count):
            if (self.children[node, 0]
                == self.children[node, 1]
                == Categorical_Tree.LEAF):
                continue
            else:
                importances[self.feature[node]] += method(node)

        normalizer = np.sum(importances)

        if normalizer > 0.0:
            # Avoid dividing by zero (e.g., when root is pure)
            importances /= normalizer

        return importances









 
