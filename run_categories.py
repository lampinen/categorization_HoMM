import numpy as np
import tensorflow as tf

from itertools import combinations

from HoMM import HoMM_model
from HoMM.configs import default_run_config, default_architecture_config
import category_tasks 

run_config = default_run_config.default_run_config
run_config.update({
    "output_dir": "results/",
    
    "base_train_tasks": [], 
    "base_eval_tasks": [], 

    "meta_class_train": ["is_basic_rule_shape", "is_basic_rule_color", "is_basic_rule_size", "is_relevant_shape", "is_relevant_color", "is_relevant_size", "is_OR", "is_AND", "is_XOR"],
    "meta_class_eval": [],
    "meta_map_train": ["NOT",
                       # color switching (a subset)
                       "switch_color_red~blue", "switch_color_blue~red", "switch_color_blue~yellow", "switch_color_yellow~blue", "switch_color_red~green", "switch_color_green~red", "switch_color_yellow~green", "switch_color_green~yellow", "switch_color_green~pink", "switch_color_pink~green", "switch_color_ocean~cyan", "switch_color_green~purple", "switch_color_purple~red", "switch_color_cyan~pink", "switch_color_pink~ocean", "switch_color_ocean~pink", 
                       # shape_switching (all except holdouts)
                       "switch_shape_triangle~square", "switch_shape_triangle~plus", "switch_shape_triangle~circle", "switch_shape_square~triangle", "switch_shape_square~plus", "switch_shape_square~circle", "switch_shape_plus~triangle", "switch_shape_plus~square", "switch_shape_circle~triangle", "switch_shape_circle~square",
                       # size switching (all except holdouts)
                       "switch_size_16~24", "switch_size_24~16", "switch_size_24~32", "switch_size_32~24",
                       ],
    "meta_map_eval": ["switch_color_yellow~purple", #"switch_color_purple~yellow",
                      "switch_shape_plus~circle", #"switch_shape_circle~plus",
                      "switch_size_16~32", #"switch_size_32~16", 
                      ],

    "refresh_mem_buffs_every": 100000000,  # no refreshing for us
})


architecture_config = default_architecture_config.default_architecture_config
architecture_config.update({
   "input_shape": [91, 91, 3],
   "output_shape": [1],

    "IO_num_hidden": 256,
    "M_num_hidden": 512,
    "H_num_hidden": 512,
    "z_dim": 512,
    "F_num_hidden": 32,
    "optimizer": "RMSProp",

    "meta_batch_size": 40,
    "meta_holdout_size": 20,

    "memory_buffer_size": 100,
})


# architecture 
def vision(processed_input, z_dim, reuse=False):
    vh = processed_input
    with tf.variable_scope("vision", reuse=reuse):
        for num_filt, kernel, stride in [[32, 4, 2],
                                         [64, 4, 2],
                                         [64, 2, 1]]:
            vh = slim.conv2d(vh,
                             num_outputs=num_filt,
                             kernel_size=kernel,
                             stride=stride,
                             padding="VALID",
                             activation_fn=tf.nn.relu)
            print(vh)
        vh = slim.flatten(vh)
        vision_out = slim.fully_connected(vh, z_dim,
                                          activation_fn=None)
    return vision_out


class category_HoMM_model(HoMM_model.HoMM_model):
    def __init__(self, run_config=None):
        super(category_HoMM_model, self).__init__(
            architecture_config=architecture_config, run_config=run_config,
            input_processor=lambda x: vision(x, architecture_config["z_dim"],
            base_loss=))

    def _pre_build_calls(self):
        # have one task in which each attribute value is "seen"
        basic_shape_tasks = [category_tasks.basic_rule("shape", [x]) for x in category_tasks.BASE_SHAPES]
        basic_color_tasks = [category_tasks.basic_rule("color", [x]) for x in category_tasks.BASE_COLORS.keys()]
        basic_size_tasks = [category_tasks.basic_rule("size", [x]) for x in category_tasks.BASE_SIZES]
        run_config["base_train_tasks"] += basic_shape_tasks + basic_color_tasks + basic_size_tasks 

        # and sampling of subsets trained and held out, but sampled to leave some interesting eval tasks 
        color_pair_tasks = [category_tasks.basic_rule("color", [x, y]) for x, y in combinations(category_tasks.BASE_COLORS.keys(), 2)]  
        train_color_pair_tasks = [x for x in color_pair_tasks if x.accepted_list not in [["red", "green"], ["blue", "yellow"], ["pink", "cyan"], ["purple", "ocean"]]
        run_config["base_train_tasks"] += train_color_pair_tasks 

        train_shape_pair_tasks = [category_tasks.basic_rule("shape", ["triangle", "square"]), category_tasks.basic_rule("shape", ["triangle", "plus"]), category_tasks.basic_rule("shape", ["square", "plus"]), category_tasks.basic_rule("shape", ["square", "circle"]), category_tasks.basic_rule("shape", ["plus", "circle"])]
        run_config["base_train_tasks"] += train_shape_pair_tasks 
        run_config["base_train_tasks"] += [category_tasks.basic_rule("size", [16, 24]), category_tasks.basic_rule("size", [16, 32])]

        # and eval tasks that target the meta-mappings, especially held-out ones:
        run_config["base_eval_tasks"] += [x for x in color_pair_tasks if x not in train_color_pair_tasks]
        run_config["base_eval_tasks"] += [category_tasks.basic_rule("shape", ["triangle", "circle"])] 
        run_config["base_eval_tasks"] += [category_tasks.basic_rule("size", [32, 24])] 

        # now feature-conjunctive tasks, part random, with targeted holdouts 
        colors = category_tasks.BASE_COLORS.keys()
        shapes = category_tasks.BASE_SHAPES
        sizes = category_tasks.BASE_SIZES
        rules = ["AND", "OR", "XOR"]
        sc_composite_tasks = [category_tasks.composite_rule(
            r,
            category_tasks.basic_rule("shape", [s]),
            category_tasks.basic_rule("color", [c])) for s in shapes for c in colors for r in rules]

        ssz_composite_tasks = [category_tasks.composite_rule(
            r,
            category_tasks.basic_rule("shape", [s]),
            category_tasks.basic_rule("size", [sz])) for s in shapes for sz in sizes for r in rules]

        csz_composite_tasks = [category_tasks.composite_rule(
            r,
            category_tasks.basic_rule("color", [c]),
            category_tasks.basic_rule("size", [sz])) for sz in sizes for c in colors for r in rules]

        train_composite_tasks = [
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["circle"]),
                category_tasks.basic_rule("color", ["yellow"])),
            category_tasks.composite_rule(
                "OR",
                category_tasks.basic_rule("shape", ["circle"]),
                category_tasks.basic_rule("color", ["yellow"])),
            category_tasks.composite_rule(
                "XOR",
                category_tasks.basic_rule("shape", ["circle"]),
                category_tasks.basic_rule("color", ["yellow"])),
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["plus"]),
                category_tasks.basic_rule("color", ["purple"])),
            category_tasks.composite_rule(
                "OR",
                category_tasks.basic_rule("shape", ["plus"]),
                category_tasks.basic_rule("color", ["purple"])),
            category_tasks.composite_rule(
                "XOR",
                category_tasks.basic_rule("shape", ["plus"]),
                category_tasks.basic_rule("color", ["purple"])),
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["circle"]),
                category_tasks.basic_rule("size", ["16"])),
            category_tasks.composite_rule(
                "OR",
                category_tasks.basic_rule("shape", ["square"]),
                category_tasks.basic_rule("size", ["16"])),
            category_tasks.composite_rule(
                "XOR",
                category_tasks.basic_rule("shape", ["triangle"]),
                category_tasks.basic_rule("size", ["16"])),
            ]
    
        eval_composite_tasks = [
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["circle"]),
                category_tasks.basic_rule("color", ["purple"])),
            category_tasks.composite_rule(
                "OR",
                category_tasks.basic_rule("shape", ["circle"]),
                category_tasks.basic_rule("color", ["purple"])),
            category_tasks.composite_rule(
                "XOR",
                category_tasks.basic_rule("shape", ["circle"]),
                category_tasks.basic_rule("color", ["purple"])),
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["circle"]),
                category_tasks.basic_rule("size", ["32"])),
            category_tasks.composite_rule(
                "OR",
                category_tasks.basic_rule("shape", ["square"]),
                category_tasks.basic_rule("size", ["32"])),
            category_tasks.composite_rule(
                "XOR",
                category_tasks.basic_rule("shape", ["triangle"]),
                category_tasks.basic_rule("size", ["32"])),
            ]

        composite_tasks = [x for x in composite_tasks if x not in eval_composite_tasks and x not in train_composite_tasks]
        np.random.shuffle(composite_tasks)

        train_composite_tasks += composite_tasks[:30]
        eval_composite_tasks += composite_tasks[30:60]

        run_config["base_train_tasks"] += train_composite_tasks 
        run_config["base_eval_tasks"] += eval_composite_tasks

        # and some negations
        train_negations = []
        eval_negations = []
        for x in basic_shape_tasks + basic_color_tasks + basic_size_tasks:
            if x.accepted_list in [set(["purple"]), set(["circle"]), set([32])]:
                eval_negations.append(category_tasks.negated(x))
            else:
                train_negations.append(category_tasks.negated(x))

        negated_train_color_pair_tasks = [category_tasks.negated(x) for x in train_color_pair_tasks]
        np.random.shuffle(negated_train_color_pair_tasks)
        train_negations += negated_train_color_pair_tasks[:-5]
        eval_negations += negated_train_color_pair_tasks[-5:]

        negated_train_shape_pair_tasks = [category_tasks.negated(x) for x in train_shape_pair_tasks]
        np.random.shuffle(negated_train_shape_pair_tasks)
        train_negations += negated_train_shape_pair_tasks[:-2]
        eval_negations += negated_train_shape_pair_tasks[-2:]

        train_negations.append(negated(category_tasks.basic_rule("size", [16, 24])))
        eval_negations.append(negated(category_tasks.basic_rule("size", [16, 32])))

        negated_train_composite_tasks = [category_tasks.negated(x) for x in train_composite_tasks]
        np.random.shuffle(negated_train_composite_tasks)
        train_negations += negated_train_composite_tasks[:-5]
        eval_negations += negated_train_composite_tasks[-5:]

        run_config["base_train_tasks"] += train_negations
        run_config["base_eval_tasks"] += eval_negations

        self.base_train_tasks = run_config["base_train_tasks"]
        self.base_eval_tasks = run_config["base_eval_tasks"]

        self.meta_class_train_tasks = run_config["meta_class_train_tasks"]
        self.meta_class_eval_tasks = run_config["meta_class_eval_tasks"]
        self.meta_map_train_tasks = run_config["meta_map_train_tasks"]
        self.meta_map_eval_tasks = run_config["meta_map_eval_tasks"]

        # set up the meta pairings 
        self.meta_pairings = category_tasks.get_meta_pairings(
            base_train_tasks=self.base_train_tasks,
            base_eval_tasks=self.base_eval_tasks,
            meta_class_train_tasks=self.meta_class_train_tasks,
            meta_class_eval_tasks=self.meta_class_eval_tasks,
            meta_map_train_tasks=self.meta_map_train_tasks,
            meta_map_eval_tasks=self.meta_map_eval_tasks) 

    def fill_buffers(self, num_data_points=1):
        del num_data_points  # don't actually use in this version
        this_tasks = self.base_train_tasks + self.base_eval_tasks 
        for t in this_tasks:
            buff = self.memory_buffers[polynomials.stringify_polynomial(t)]
            x_data = np.zeros([num_data_points] + self.architecture_config["input_shape"])
            y_data = np.zeros([num_data_points] + self.architecture_config["output_shape"])
            for point_i in range(num_data_points):
                point = t.family.sample_point(val_range=self.run_config["point_val_range"])
                x_data[point_i, :] = point
                y_data[point_i, :] = t.evaluate(point)
            buff.insert(x_data, y_data)

    def build_feed_dict(self, task, lr=None, fed_embedding=None,
                        call_type="base_standard_train"):
        """Build a feed dict.

        Mildly overridden to sample examples more carefully into guess for base.
        """
        feed_dict = {}
        original_call_type = call_type

        base_or_meta, call_type, train_or_eval = call_type.split("_")

        if base_or_meta == "base":
            task_name, memory_buffer, task_index = self.base_task_lookup(task)
            inputs, outputs = self.sample_from_memory_buffer(memory_buffer)
            feed_dict[self.base_input_ph] = inputs
            feed_dict[self.base_target_ph] = outputs
            mask = np.zeros(len(outputs), dtype=np.bool)
            pos_indices = np.argwhere(outputs) 
            neg_indices = np.argwhere(np.logical_not(outputs)) 
            num_pos = len(pos_indices)
            num_neg = len(neg_indices)
            np.shuffle(pos_indices)
            np.shuffle(neg_indices)
            elif num_pos > num_neg:
                small_set_size = min(num_neg//2, self.meta_batch_size//2)
                pos_indices = pos_indices[:self.meta_batch_size - small_set_size]
                neg_indices = neg_indices[:small_set_size]
            else:
                small_set_size = min(num_pos//2, self.meta_batch_size//2)
                pos_indices = pos_indices[:small_set_size]
                neg_indices = pos_indices[:self.meta_batch_size - small_set_size]
            indices = pos_indices + neg_indices 
            mask[indices] = True
            feed_dict[self.guess_input_mask_ph] = mask 
        else:   # meta dicts are the same
            feed_dict = super(grids_HoMM_agent, self).build_feed_dict(
                task=task, lr=lr, fed_embedding=fed_embedding, call_type=original_call_type)

        if call_type == "fed":
            if len(fed_embedding.shape) == 1:
                fed_embedding = np.expand_dims(fed_embedding, axis=0)
            feed_dict[self.feed_embedding_ph] = fed_embedding
        elif call_type == "lang":
            feed_dict[self.language_input_ph] = self.intify_task(task_name)
        if call_type == "cached" or self.architecture_config["persistent_task_reps"]:
            feed_dict[self.task_index_ph] = [task_index]

        if train_or_eval == "train":
            feed_dict[self.lr_ph] = lr
            feed_dict[self.keep_prob_ph] = self.tkp
            if call_type == "lang":
                feed_dict[self.lang_keep_prob_ph] = self.lang_keep_prob
        else:
            feed_dict[self.keep_prob_ph] = 1.
            if call_type == "lang":
                feed_dict[self.lang_keep_prob_ph] = 1.

        return feed_dict


## running stuff
for run_i in range(run_config["num_runs"]):
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    run_config["this_run"] = run_i

    model = category_HoMM_model(run_config=run_config)
    model.run_training()

    tf.reset_default_graph()
