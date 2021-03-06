from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.ndimage

import matplotlib.pyplot as plt

BASE_SHAPES = ["triangle", "square", 
               "plus", "circle", "tee",
               "inverseplus", "emptysquare",
               "emptytriangle"]
BASE_COLORS = {
    "red": (1., 0., 0.),
    "green": (0., 1., 0.),
    "blue": (0., 0., 1.),
    "yellow": (1., 1, 0.),
    "pink": (1., 0.4, 1.),
    "cyan": (0., 1., 1.),
    "purple": (0.3, 0., 0.5),
    "ocean": (0.1, 0.4, 0.5),
    #"orange": (1., 0.6, 0.),
    #"forest": (0., 0.5, 0.),
}
BASE_SIZES = ["16", "24", "32"]

RENDER_SIZE = 50 

RANDOM_ANGLE_RANGE = 20

BASE_COLORS = {n: np.array(c, dtype=np.float32) for n, c in BASE_COLORS.items()} 

def _render_plain_shape(name, size):
    """Shape without color dimension."""
    size = int(size)
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
    elif name == "tee":
        shape[:, size // 2 - size // 6: size // 2 + size //6 + 1] = 1.
        shape[:size // 3, :] = 1.
    elif name == "inverseplus":
        shape[:, :] = 1.
        shape[:, size // 2 - size // 6: size // 2 + size //6 + 1] = 0.
        shape[size // 2 - size // 6: size // 2 + size //6 + 1, :] = 0.
    elif name == "emptysquare":
        shape[:, :size // 8] = 1.
        shape[:, -size // 8:] = 1.
        shape[:size // 8, :] = 1.
        shape[-size // 8:, :] = 1.
    elif name == "emptytriangle":
        for i in range(size):
            for j in range(size):
                if np.abs(np.abs(j - size // 2) - np.abs(i // 2)) < size // 10:
                    shape[i, j] = 1.
        shape[-size // 10:, :] = 1.
    return shape


_base_templates = {(s, sz): _render_plain_shape(s, sz) for s in BASE_SHAPES for sz in BASE_SIZES} 

def render_uncolored_shape(name, size):
    "Shape without color dimension, at random rotation and position."
    template = _base_templates[(name, size)]
    angle = np.random.randint(-RANDOM_ANGLE_RANGE, RANDOM_ANGLE_RANGE)
    shape = scipy.ndimage.rotate(template, angle, order=1)
    new_size = shape.shape
    image = np.zeros([RENDER_SIZE, RENDER_SIZE], np.float32)
    offset_x = np.random.randint(0, RENDER_SIZE - new_size[0])
    offset_y = np.random.randint(0, RENDER_SIZE - new_size[1])
    image[offset_x:offset_x + new_size[0], 
          offset_y:offset_y + new_size[1]] = shape
    return image 


class categorization_instance(object):
    def __init__(self, shape, color, size):
        self.shape = shape
        self.color = color
        self.raw_color = BASE_COLORS[self.color]
        self.size = size

    def render(self):
        plain_image = render_uncolored_shape(self.shape, self.size) 
        image = plain_image[:, :, None] * self.raw_color[None, None, :]
        return image

    def __str__(self):
        return "{}_{}_{}".format(self.shape, self.color, self.size)

    def __eq__(self, other):
        return isinstance(other, categorization_instance) and self.shape == other.shape and self.color == other.color and self.size == other.size

    def differs_by_one(self, other): 
        if not isinstance(other, categorization_instance):
            raise ValueError("Categorization instances can only be compared to other categorization instances")

        matches = [self.shape == other.shape, self.color == other.color, self.size == other.size]
        return sum(matches) == 2


class basic_rule(object):
    def __init__(self, attribute_type, accepted_list):
        self.attribute_type = attribute_type
        self.accepted_list = set(accepted_list)
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
        return "{}={}".format(self.attribute_type, "&".join(sorted(self.accepted_list)))

    def __eq__(self, other):
        return isinstance(other, basic_rule) and self.attribute_type == other.attribute_type and self.accepted_list == other.accepted_list


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
        return "{}(({})&({}))".format(self.rule_type, 
                                   str(self.rule_a),
                                   str(self.rule_b))

    def __eq__(self, other):
        if not isinstance(other, composite_rule):
            return False
        return self.rule_type == other.rule_type and ((self.rule_a == other.rule_a and self.rule_b == other.rule_b) or (self.rule_a == other.rule_b and self.rule_b == other.rule_a))


def _and_to_or_inner(rule):
    if isinstance(rule, basic_rule):
        return rule
    if rule.rule_type == "AND":
        new_rule_type = "OR"
    elif rule.rule_type == "OR":
        new_rule_type = "AND"
    else: 
        new_rule_type = rule.rule_type
    result_a = _and_to_or_inner(rule.rule_a)
    result_b = _and_to_or_inner(rule.rule_b)
    return composite_rule(new_rule_type, result_a, result_b)


def and_to_or(rule):
    if isinstance(rule, basic_rule):
        return None
    
    result_a = None
    result_b = None
    if rule.rule_type not in ["AND", "OR"]:
        if isinstance(rule.rule_a, composite_rule):
            result_a = and_to_or(rule.rule_a)
        if isinstance(rule.rule_b, composite_rule):
            result_b = and_to_or(rule.rule_b)
        if result_a is None and result_b is None:
            return None
        else:
            new_rule_type = rule.rule_type
        return composite_rule(new_rule_type, result_a, result_b)

    if rule.rule_type == "AND":
        new_rule_type = "OR"
    elif rule.rule_type == "OR":
        new_rule_type = "AND"

    if result_a is None:
        result_a = _and_to_or_inner(rule.rule_a)
    if result_b is None:
        result_b = _and_to_or_inner(rule.rule_b)
    return composite_rule(new_rule_type, result_a, result_b)


def negated(rule):
    if isinstance(rule, basic_rule):
        if rule.attribute_type == "shape":
            new_accepted_list = [x for x in BASE_SHAPES if x not in rule.accepted_list] 
        elif rule.attribute_type == "color":
            new_accepted_list = [x for x in BASE_COLORS.keys() if x not in rule.accepted_list] 
        else: 
            new_accepted_list = [x for x in BASE_SIZES if x not in rule.accepted_list] 
        return basic_rule(attribute_type=rule.attribute_type,
                          accepted_list=new_accepted_list)
    else:
        result_a = negated(rule.rule_a)
        result_b = negated(rule.rule_b)
        if rule.rule_type == "OR":
            new_rule_type = "AND"
        elif rule.rule_type == "AND":
            new_rule_type = "OR"
        else:
            new_rule_type = "XOR"
            result_b = rule.rule_b  # NXOR(A, B) = XOR(!A, B)
    return composite_rule(new_rule_type, result_a, result_b)
        

def _switch_basic_attribute_inner(rule, target_attribute_type, pairs):
    if isinstance(rule, basic_rule):
        if rule.attribute_type == target_attribute_type:
            new_accepted_list = []
            for x in rule.accepted_list:
                if x in pairs:
                    new_accepted_list.append(pairs[x])
                else:
                    new_accepted_list.append(x)
            return basic_rule(attribute_type=rule.attribute_type,
                              accepted_list=new_accepted_list)
        else:
            return rule 
    else:
        result_a = _switch_basic_attribute_inner(rule.rule_a)
        result_b = _switch_basic_attribute_inner(rule.rule_b)
        return composite_rule(rule.rule_type, result_a, result_b) 


def switch_basic_attribute(rule, target_attribute_type, pairs):
    if isinstance(rule, basic_rule):
        if rule.attribute_type == target_attribute_type:
            new_accepted_list = []
            for x in rule.accepted_list:
                if x in pairs:
                    new_accepted_list.append(pairs[x])
                else:
                    new_accepted_list.append(x)
            return basic_rule(attribute_type=rule.attribute_type,
                              accepted_list=new_accepted_list)
        else:
            return None
    else:
        result_a = switch_basic_attribute(rule.rule_a, target_attribute_type, 
                                           pairs)
        result_b = switch_basic_attribute(rule.rule_b, target_attribute_type,
                                           pairs)
        if result_a is None and result_b is None:
            return None
        if result_a is None:
            result_a = _switch_basic_attribute_inner(rule.rule_a, 
                                                     target_attribute_type,
                                                     pairs)
        if result_b is None:
            result_b = _switch_basic_attribute_inner(rule.rule_b,
                                                     target_attribute_type,
                                                     pairs)
        return composite_rule(rule.rule_type, result_a, result_b)


def construct_task_instance_dict(task, instances):
    """Construct a dict classifying the instances for this task.

    Instances are classified into positive, contrasting negative examples
    paired with the positive example they contrast with, or other negative."""
    examples = {"positive": [], "contrasting": [], "other": [], "all_negative": []}
    for inst in instances:
        label = task.apply(inst)
        if label:
            examples["positive"].append(inst)
        else:
            examples["all_negative"].append(inst)
    examples["contrasting"] = [[] for _ in examples["positive"]]

    for inst in examples["all_negative"]:
        fits_as_contrasting = False
        for i, pos_inst in enumerate(examples["positive"]):
            if inst.differs_by_one(pos_inst):
                examples["contrasting"][i].append(inst)
                fits_as_contrasting = True
        if not fits_as_contrasting:
            examples["other"].append(inst)

    return examples

def get_meta_pairings(base_train_tasks, base_eval_tasks, meta_class_train_tasks, meta_class_eval_tasks,
                      meta_map_train_tasks, meta_map_eval_tasks):
    """Gets which tasks map to which other tasks under the meta mappings."""
    all_meta_tasks = meta_class_train_tasks + meta_class_eval_tasks + meta_map_train_tasks + meta_map_eval_tasks
    meta_pairings = {mt: {"train": [], "eval": []} for mt in all_meta_tasks}
    for mt in all_meta_tasks:
        
        if mt == "NOT":
            meta_mapping = lambda rule: negated(rule)
        elif mt == "AND_to_OR":
            meta_mapping = lambda rule: and_to_or(rule)
        elif mt[:7] == "switch_":
            commands = mt.split("_")
            target_attribute_type = commands[1]
            pairs = dict([x.split("~") for x in commands[2:]])
            meta_mapping = lambda rule: switch_basic_attribute(
                rule, target_attribute_type, pairs)
        elif mt[:3] == "is_":  # meta classification
            classification_type = mt[3:]
            if classification_type == "composite":
                meta_mapping = lambda rule: isinstance(rule, composite_rule)
            elif classification_type[:10] == "basic_rule":
                attribute_type = classification_type[11:]  # skip underscore
                meta_mapping = lambda rule: isinstance(rule, basic_rule) and rule.attribute_type == attribute_type 
            elif classification_type[:8] == "relevant":
                attribute_type = classification_type[9:]
                def meta_mapping(rule):
                    if isinstance(rule, basic_rule):
                        return rule.attribute_type == attribute_type 
                    else: 
                        return meta_mapping(rule.rule_a) or meta_mapping(rule.rule_b)
            else:  # type of composite rule
                meta_mapping = lambda rule: isinstance(rule, composite_rule) and rule.rule_type == classification_type 
        else: 
            raise ValueError("Unrecognized meta task: {}".format(mt))
        
        if mt[:3] == "is_":
            for curr_tasks, train_or_eval in zip([base_train_tasks,
                                                  base_eval_tasks],
                                                 ["train", "eval"]):
                for task in curr_tasks:
                    res = meta_mapping(task)
                    meta_pairings[mt][train_or_eval].append((str(task), 1. * res))

        else:
            for task in base_train_tasks:
                res = meta_mapping(task)
                if res is not None and res != task:
                    if res in base_train_tasks:
                        meta_pairings[mt]["train"].append((str(task), str(res)))
                    elif res in base_eval_tasks:
                        meta_pairings[mt]["eval"].append((str(task), str(res)))

    return meta_pairings


if __name__ == "__main__":
    tasks = [basic_rule("shape", ["triangle", "circle"]),
             basic_rule("color", ["red"]),
             basic_rule("size", ["24"])] 
    tasks.append(composite_rule("OR", tasks[1], tasks[2]))
    tasks.append(composite_rule("XOR", tasks[1], tasks[2]))
    tasks.append(composite_rule("AND", tasks[1], tasks[2]))
    tasks.append(composite_rule("AND", tasks[-2], tasks[-1]))  # always false

    for sz in BASE_SIZES[1:]:
        for s in ["triangle", "inverseplus", "emptysquare", "circle"]:
            for c in ["red", "yellow", "purple", "cyan"]:
                for sample_i in range(2):
                    inst = categorization_instance(s, c, sz)
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(3, 3)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    plt.imshow(inst.render(), aspect='auto')
                    plt.savefig("stimulus_renders/%s_%s_%s_%i.png" % (sz, c, s, sample_i))
                    plt.close()

    for t in tasks:
        print(t)
        for s in BASE_SHAPES:
            for sz in BASE_SIZES:
                for c in BASE_COLORS.keys():
                    inst = categorization_instance(s, c, sz)
                plt.imshow(inst.render())
                plt.show()

    print(negated(tasks[1]))
    print(negated(tasks[3]))
    print(and_to_or(tasks[1]))
    print(and_to_or(tasks[3]))
    print(switch_basic_attribute(tasks[1], "color", {"red": "blue"}))
    print(switch_basic_attribute(tasks[3], "color", {"red": "blue"}))
    print()
    pairs = get_meta_pairings(tasks[:-1], tasks[-1:], ["is_basic_rule_color", "is_basic_rule_size", "is_relevant_color", "is_relevant_shape", "is_OR", "is_XOR"], [], ["NOT", "AND_to_OR", "switch_color_red~blue", "switch_shape_triangle~square_circle~plus"], []) 
    for k, v in pairs.items():
        print(k)
        print(v)
        
