from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

BASE_SHAPES = ["triangle", "square", 
               "plus", "circle"]
BASE_COLORS = {
    "red": (1., 0., 0.),
    "green": (0., 1., 0.),
    "blue": (0., 0., 1.),
#    "yellow": (1., 1, 0.),
#    "pink": (1., 0.4, 1.),
#    "cyan": (0., 1., 1.),
#    "purple": (0.3, 0., 0.5),
#    "ocean": (0.1, 0.4, 0.5),
    #"orange": (1., 0.6, 0.),
    #"forest": (0., 0.5, 0.),
}
BASE_SIZES = [16, 24]#, 32]

RENDER_SIZE = 32 

BASE_COLORS = {n: np.array(c, dtype=np.float32) for n, c in BASE_COLORS.items()} 

def _render_plain_shape(name, size):
    """Shape without color dimension"""
    shape = np.zeros([size, size], np.float32)
    if name == "square":
        shape[:, :] = 1.
    elif name == "circle":
        for i in range(size):
            for j in range(size):
                if np.square(i + 0.5 - size // 2) + np.square(j + 0.5 - size // 2) < np.square(size // 2):
                    shape[i, j] = 1.
    elif name == "triangle":
        for i in range(size):
            for j in range(size):
                if np.abs(j - size // 2) - np.abs(i // 2) < 1:
                    shape[i, j] = 1.
    elif name == "plus":
        shape[:, size // 2 - size // 6: size // 2 + size //6 + 1] = 1.
        shape[size // 2 - size // 6: size // 2 + size //6 + 1, :] = 1.
    image = np.zeros([RENDER_SIZE, RENDER_SIZE], np.float32)
    offset = (RENDER_SIZE - size) // 2
    image[offset:offset + size, offset:offset + size] = shape
    return image 


class categorization_instance(object):
    def __init__(self, shape, color, size):
        self.shape = shape
        self.color = color
        self.size = size
        plain_image = _render_plain_shape(shape, size) 
        raw_color = BASE_COLORS[color]
        self.image = plain_image[:, :, None] * raw_color[None, None, :]

    def __str__(self):
        return "{}_{}_{}".format(self.shape, self.color, self.size)


class basic_rule(object):
    def __init__(self, attribute_type, accepted_list):
        self.attribute_type = attribute_type
        self.accepted_list = accepted_list
        self.compiled_rule = self._compile_rule_func()

    def apply(self, instance):
        return self.compiled_rule(instance)

    def _compile_rule_func(self):
        if self.attribute_type == "shape":
            return lambda inst: inst.shape in self.accepted_list
        elif self.attribute_type == "color":
            return lambda inst: inst.color in self.accepted_list
        else: 
            return lambda inst: inst.size in self.accepted_list

    def __str__(self):
        return "{}:{}".format(self.attribute_type, self.accepted_list)


def XOR(a, b):
    return (a + b) % 2 != 0


class composite_rule(object):
    def __init__(self, rule_type, rule_a, rule_b): 
        self.rule_type = rule_type
        self.rule_a = rule_a
        self.rule_b = rule_b
        self.compiled_rule = self._compile_rule_func()

    def _compile_rule_func(self): 
        if self.rule_type == "OR":
            return lambda inst: self.rule_a.compiled_rule(inst) or self.rule_b.compiled_rule(inst)
        elif self.rule_type == "AND": 
            return lambda inst: self.rule_a.compiled_rule(inst) and self.rule_b.compiled_rule(inst)
        elif self.rule_type == "XOR":
            return lambda inst: XOR(self.rule_a.compiled_rule(inst), self.rule_b.compiled_rule(inst))

    def apply(self, instance):
        return self.compiled_rule(instance)

    def __str__(self):
        return "{}({}, {})".format(self.rule_type, 
                                   str(self.rule_a),
                                   str(self.rule_b))


if __name__ == "__main__":
    tasks = [basic_rule("shape", ["triangle", "circle"]),
             basic_rule("color", ["red"]),
             basic_rule("size", [24])] 
    tasks.append(composite_rule("OR", tasks[1], tasks[2]))
    tasks.append(composite_rule("XOR", tasks[1], tasks[2]))
    tasks.append(composite_rule("AND", tasks[1], tasks[2]))
    tasks.append(composite_rule("AND", tasks[-2], tasks[-1]))  # always false
        
    for t in tasks:
        print(t)
        for s in BASE_SHAPES:
            for sz in BASE_SIZES:
                for c in BASE_COLORS.keys():
                    inst = categorization_instance(s, c, sz)
                    print(inst, t.apply(inst))
#                plt.imshow(inst.image)
#                plt.show()
