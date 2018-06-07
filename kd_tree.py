

# a data feature - n-dimensional vector
class Feature:
    def __init__(self, _id: int, _num_dims: int):
        self.data = [0.0] * _num_dims
        self.id = _id

    def __repr__(self):
        str_list = [str(item) for item in self.data]
        if self.id >= 0:
            return 'idx ' + str(self.id) + ' = (' + ','.join(str_list) + ')'
        else:
            return '(' + ','.join(str_list) + ')'


# a node in the tree
class KDNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.variance_dimension_index = -1.0  # the index of the most variant dimension
        self.median_value = 0.0  # the median to compare with of that dimension
        self.id_list = []  # ids of the final values held by this node


# for searching
class MQNode:
    def __init__(self, node: KDNode, dist: float):
        self.node = node
        self.dist = dist


# the kd-tree
class KDTree:
    def __init__(self, _num_dims: int, _vector_list):
        self.num_dims = _num_dims
        self.vector_map = dict()
        for vector in _vector_list:
            self.vector_map[vector.id] = vector
        self._setup()

    # perform a find in the tree starting at the root
    def find(self, vector):
        mq = [MQNode(self.root, 0.0)]
        min_dist = 1_000_000.0
        best_node = -1
        while len(mq) > 0:
            mq.sort(key=lambda x: x.dist)
            tmp_mq_node = mq.pop(0)
            tmp_kd_node = tmp_mq_node.node
            tmp_variance_dimension_index = tmp_kd_node.variance_dimension_index
            tmp_median_value = tmp_kd_node.median_value

            # if leaf node, search that node.
            if tmp_variance_dimension_index == -1:
                for id in tmp_kd_node.id_list:
                    tmp_vector = self.vector_map[id]
                    dist = 0.0
                    for i in range(0, self.num_dims):
                        delta = vector.data[i] - tmp_vector.data[i]
                        dist += delta * delta
                    if dist < min_dist:
                        min_dist = dist
                        best_node = id
            else:
                if vector.data[tmp_variance_dimension_index] < tmp_median_value:
                    mq.append(MQNode(tmp_kd_node.left, 0.0))
                    mq.append(MQNode(tmp_kd_node.right, tmp_median_value - vector.data[tmp_variance_dimension_index]))
                else:
                    mq.append(MQNode(tmp_kd_node.right, 0.0))
                    mq.append(MQNode(tmp_kd_node.left, vector.data[tmp_variance_dimension_index] - tmp_median_value))

        return best_node

    # get a feature using the id
    def get_feature_from_id(self, id: int):
        return self.vector_map[id]

    # setup the tree root and divisions
    def _setup(self):
        self.root = KDNode()
        self.root.id_list = [id for id in self.vector_map.keys()]
        self._divide_kd_node(self.root)

    # divide the ids of a node into two across the most variant median of the n-dimensions
    def _divide_kd_node(self, node):
        if len(node.id_list) <= 1:
            return
        self._set_partition(node)
        vdi = node.variance_dimension_index
        median = node.median_value

        # move data into left and right depending on median of the dimension
        left = KDNode()
        right = KDNode()
        for id in node.id_list:
            if self.vector_map[id].data[vdi] < median:
                left.id_list.append(id)
            else:
                right.id_list.append(id)

        if len(left.id_list) == 0 or len(right.id_list) == 0:
            node.variance_dimension_index = -1
            return

        node.id_list = []
        node.left = left
        node.right = right

        self._divide_kd_node(left)
        self._divide_kd_node(right)

    # calculate median for the most variant index of all the data represented in node
    def _set_partition(self, node):
        mean_list = [0.0] * self.num_dims
        mean2_list = [0.0] * self.num_dims
        for id in node.id_list:
            for i in range(0, self.num_dims):
                x = self.vector_map[id].data[i]
                mean_list[i] += x
                mean2_list[i] += x * x

        variance_list = [0.0] * self.num_dims
        size = len(node.id_list)
        for j in range(0, self.num_dims):
            mean_list[j] /= size
            mean2_list[j] /= size
            v = mean_list[j]
            variance_list[j] = mean2_list[j] - v * v

        # ki : the index of the key with the largest variance
        max_value = max(variance_list)
        node.variance_dimension_index = variance_list.index(max_value)
        median = [v.data[node.variance_dimension_index] for v in self.vector_map.values()]
        median.sort()
        node.median_value = median[len(median) // 2]


##########################################################################################################
# test it

num_dims = 5
num_items = 32
test_vector_list = []
for i in range(0, num_items):
    f = Feature(i + 1, num_dims)
    for j in range(0, num_dims):
        f.data[j] = (i >> j & 1) * 10.0
    test_vector_list.append(f)

# print the test vectors for info
print("kd-tree content")
for vector in test_vector_list:
    print(vector.__repr__())
print()

# create the tree from these vectors
kd_tree = KDTree(num_dims, test_vector_list)

# create a feature to look for
find_feature = Feature(-1, num_dims)
find_feature.data = [2.0, 2.0, 2.0, 8.0, 8.0]
print("search for " + find_feature.__repr__())

# search!
closest_feature_id = kd_tree.find(find_feature)
print("found " + str(kd_tree.get_feature_from_id(closest_feature_id)) + " to be the closest vector to " + find_feature.__repr__())

