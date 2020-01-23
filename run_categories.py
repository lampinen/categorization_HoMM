import numpy as np
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim

from itertools import combinations

from HoMM import HoMM_model
from HoMM.configs import default_run_config, default_architecture_config
import category_tasks 

run_config = default_run_config.default_run_config
run_config.update({
    "output_dir": "/mnt/fs4/lampinen/categorization_HoMM/results_132/",

    "run_offset": 0,
    "num_runs": 1,
    
    "base_train_tasks": [], 
    "base_eval_tasks": [], 

    "meta_class_train_tasks": ["is_basic_rule_shape", "is_basic_rule_color", "is_basic_rule_size", "is_relevant_shape", "is_relevant_color", "is_relevant_size", "is_OR", "is_AND", "is_XOR"],
    "meta_class_eval_tasks": [],
    "meta_map_train_tasks": [#"NOT",
                       # color switching (all except holdouts, but note some may get dropped) 
                       "switch_color_blue~pink", "switch_color_blue~purple", "switch_color_blue~yellow", "switch_color_blue~ocean", "switch_color_blue~green", "switch_color_blue~cyan", "switch_color_blue~red", "switch_color_pink~blue", "switch_color_pink~purple", "switch_color_pink~yellow", "switch_color_pink~ocean", "switch_color_pink~green", "switch_color_pink~cyan", "switch_color_pink~red", "switch_color_purple~blue", "switch_color_purple~pink", "switch_color_purple~ocean", "switch_color_purple~green", "switch_color_purple~cyan", "switch_color_purple~red", "switch_color_yellow~blue", "switch_color_yellow~pink", "switch_color_yellow~ocean", "switch_color_yellow~green", "switch_color_yellow~cyan", "switch_color_yellow~red", "switch_color_ocean~blue", "switch_color_ocean~pink", "switch_color_ocean~purple", "switch_color_ocean~yellow", "switch_color_ocean~green", "switch_color_ocean~cyan", "switch_color_ocean~red", "switch_color_green~blue", "switch_color_green~pink", "switch_color_green~purple", "switch_color_green~yellow", "switch_color_green~ocean", "switch_color_green~cyan", "switch_color_green~red", "switch_color_cyan~blue", "switch_color_cyan~pink", "switch_color_cyan~purple", "switch_color_cyan~yellow", "switch_color_cyan~ocean", "switch_color_cyan~green", "switch_color_cyan~red", "switch_color_red~blue", "switch_color_red~pink", "switch_color_red~purple", "switch_color_red~yellow", "switch_color_red~ocean", "switch_color_red~green", "switch_color_red~cyan",
                       # shape_switching (all except holdouts, but note some may get dropped)
                       "switch_shape_triangle~square", "switch_shape_triangle~plus", "switch_shape_triangle~circle", "switch_shape_square~triangle", "switch_shape_square~plus", "switch_shape_square~circle", "switch_shape_plus~triangle", "switch_shape_plus~square", "switch_shape_circle~triangle", "switch_shape_circle~square", "switch_shape_triangle~tee", "switch_shape_triangle~inverseplus", "switch_shape_triangle~emptysquare", "switch_shape_square~tee", "switch_shape_square~inverseplus", "switch_shape_square~emptysquare", "switch_shape_plus~tee", "switch_shape_plus~inverseplus", "switch_shape_plus~emptysquare", "switch_shape_circle~tee", "switch_shape_circle~inverseplus", "switch_shape_circle~emptysquare", "switch_shape_tee~inverseplus", "switch_shape_tee~emptysquare", "switch_shape_inverseplus~tee", "switch_shape_inverseplus~emptysquare", "switch_shape_emptysquare~tee", "switch_shape_emptysquare~inverseplus", "switch_shape_triangle~emptytriangle", "switch_shape_square~emptytriangle", "switch_shape_plus~emptytriangle", "switch_shape_circle~emptytriangle", "switch_shape_tee~emptytriangle", "switch_shape_inverseplus~emptytriangle", "switch_shape_emptysquare~emptytriangle", "switch_shape_emptytriangle~triangle", "switch_shape_emptytriangle~square", "switch_shape_emptytriangle~plus", "switch_shape_emptytriangle~circle", "switch_shape_emptytriangle~tee", "switch_shape_emptytriangle~inverseplus", "switch_shape_emptytriangle~emptysquare"
                       # size switching (all)
                       #"switch_size_16~24", "switch_size_24~16", "switch_size_24~32", "switch_size_32~24", "switch_size_16~32", "switch_size_32~16",
                       ],
    "meta_map_eval_tasks": ["switch_color_yellow~purple", #"switch_color_purple~yellow",
                            "switch_shape_plus~circle", #"switch_shape_circle~plus",
                           ],

    "include_size_tasks": False,
    "include_pair_tasks": False,
    "train_ext_composite_tasks": 30,  # should be sufficiently less than 54 (with current settings) to leave enough test tasks
    "meta_min_train_threshold": 8,  # minimum number of train items for a mapping, those with fewer will be removed 

    "multiplicity": 2,  # how many different renders of each object to put in memory

    "refresh_mem_buffs_every": 20,
    "eval_every": 20,
    "lr_decays_every": 400,

    "init_learning_rate": 3e-5,  # initial learning rate for base tasks
    "init_meta_learning_rate": 1e-6,  # for meta-classification and mappings

    "lr_decay": 0.8,  # how fast base task lr decays (multiplicative)
    "meta_lr_decay": 0.85,

    "min_learning_rate": 1e-8,  # can't decay past these minimum values 
#    "min_language_learning_rate": 3e-8,
    "min_meta_learning_rate": 1e-8,

    "num_epochs": 1000000,
    "include_noncontrasting_negative": False,  # if True, half of negative examples will be random
    "note": "random angle range reduced; no negation; no size meta; more meta color + shape; new shape; Mapping domain fix."
})


architecture_config = default_architecture_config.default_architecture_config
architecture_config.update({
   "input_shape": [50, 50, 3],
   "output_shape": [1],

    "IO_num_hidden": 512,
    "M_num_hidden": 1024,
    "H_num_hidden": 512,
    "z_dim": 512,
    "F_num_hidden": 128,
    "optimizer": "Adam",

    "F_weight_normalization": False,
    "F_wn_strategy": "standard",

    "F_num_hidden_layers": 3,
    "mlp_output": True,

#    "train_drop_prob": 0.5,

    "meta_batch_size": 128,
#    "meta_holdout_size": 30,

    "memory_buffer_size": 336,

    "task_weight_weight_mult": 30.,

    "vision_layers": [[64, 5, 2, False],
                      [128, 4, 2, False],
                      [256, 4, 2, False],
                      [512, 2, 2, True]],
})

if False:  # enable for language baseline
    run_config.update({
        "train_language_base": True,
        "train_base": False,
        "train_meta": False,
        "init_language_learning_rate": 3e-5,  

        "vocab": ["PAD"] + ["AND", "OR", "XOR"] + ["(", ")", "=", "&"] + ["shape", "size", "color"] + category_tasks.BASE_SIZES + category_tasks.BASE_SHAPES + list(category_tasks.BASE_COLORS.keys()),

        "output_dir": run_config["output_dir"] + "language/",  # subfolder
    })

if False:  # enable for homoiconic language-based training and meta-mapping 
    run_config.update({
        "train_language_base": True,
        "train_language_meta": True,
        "train_base": False,
        "train_meta": False,

        "init_language_learning_rate": 3e-5,  
        "init_language_meta_learning_rate": 1e-6,  
        "language_lr_decay": 0.85, 
        "vocab": ["PAD"] + ["is", "basic", "rule", "relevant", "switch"] + ["AND", "OR", "XOR"] + ["(", ")", "=", "&", "~"] + ["shape", "size", "color"] + category_tasks.BASE_SIZES + category_tasks.BASE_SHAPES + list(category_tasks.BASE_COLORS.keys()),

        "output_dir": run_config["output_dir"] + "language_HoMM/",  # subfolder
    })

if False:  # enable for persistent
    architecture_config.update({
        "persistent_task_reps": True,
        "combined_emb_guess_weight": "varied",
        "emb_match_loss_weight": 0.2,
    })


class memory_buffer(object):
    """Essentially a wrapper around numpy arrays that handles inserting and
    removing."""
    def __init__(self, length, input_shape, outcome_width):
        self.length = length
        self.num_positive = None
        self.input_buffer = np.zeros([length] + input_shape)
        self.outcome_buffer = np.zeros([length, outcome_width])

    def insert(self, input_mat, outcome_mat, num_positive):
        self.input_buffer = input_mat
        self.outcome_buffer = outcome_mat
        self.num_positive = num_positive

    def get_memories(self):
        return self.input_buffer, self.outcome_buffer, self.num_positive


# architecture 
def vision(processed_input, z_dim, IO_num_hidden, vision_layers, reuse=False):
    vh = processed_input
    with tf.variable_scope("vision", reuse=reuse):
        for num_filt, kernel, stride, mp in vision_layers:
            vh = slim.conv2d(vh,
                             num_outputs=num_filt,
                             kernel_size=kernel,
                             stride=stride,
                             padding="VALID",
                             activation_fn=tf.nn.leaky_relu)
            print(vh)
            if mp:
                vh = slim.max_pool2d(vh, [2, 2], padding="SAME")
                print(vh)
        vh = slim.flatten(vh)
        vh = slim.fully_connected(vh, IO_num_hidden,
                                  activation_fn=tf.nn.leaky_relu)
        vision_out = slim.fully_connected(vh, z_dim,
                                          activation_fn=None)
    return vision_out


def mlp_output_processor(output_embeddings, IO_num_hidden, output_size):
    with tf.variable_scope("output_processor", reuse=tf.AUTO_REUSE):
        output_hidden = slim.fully_connected(output_embeddings, IO_num_hidden,
                                             activation_fn=tf.nn.leaky_relu)
        processed_outputs = slim.fully_connected(output_hidden, output_size,
                                                 activation_fn=None)
    return processed_outputs



def xe_loss(output_logits, targets, backward_mask):  # xe only on held out examples
    mask = tf.math.logical_not(backward_mask)
    masked_logits = tf.boolean_mask(output_logits, mask)
    masked_targets = tf.boolean_mask(targets, mask)
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=masked_logits,
                                                labels=masked_targets))


class category_HoMM_model(HoMM_model.HoMM_model):
    def __init__(self, run_config=None):
        self.include_noncontrasting_negative = run_config["include_noncontrasting_negative"]
        if architecture_config["mlp_output"]:
            output_processor = lambda x: mlp_output_processor(
                x, architecture_config["IO_num_hidden"], architecture_config["output_shape"][0])
        else:
            output_processor = None
        super(category_HoMM_model, self).__init__(
            architecture_config=architecture_config, run_config=run_config,
            input_processor=lambda x: vision(x, architecture_config["z_dim"],
                                             architecture_config["IO_num_hidden"],
                                             architecture_config["vision_layers"]),
            output_processor=output_processor,
            base_loss=lambda x, y: xe_loss(x, y, self.guess_input_mask_ph))

    def _pre_build_calls(self):
        # have one task in which each attribute value is "seen"
        basic_shape_tasks = [category_tasks.basic_rule("shape", [x]) for x in category_tasks.BASE_SHAPES]
        basic_color_tasks = [category_tasks.basic_rule("color", [x]) for x in category_tasks.BASE_COLORS.keys()]
        run_config["base_train_tasks"] += basic_shape_tasks + basic_color_tasks
        if run_config["include_size_tasks"]:
            basic_size_tasks = [category_tasks.basic_rule("size", [x]) for x in category_tasks.BASE_SIZES]
            run_config["base_train_tasks"] += basic_size_tasks 

        # and sampling of subsets trained and held out, but sampled to leave some interesting eval tasks 
        if run_config["include_pair_tasks"]:
            color_pair_tasks = [category_tasks.basic_rule("color", [x, y]) for x, y in combinations(category_tasks.BASE_COLORS.keys(), 2)]  
            train_color_pair_tasks = [x for x in color_pair_tasks if x.accepted_list not in [set(["red", "green"]), set(["blue", "yellow"]), set(["pink", "cyan"]), set(["purple", "ocean"])]]
            run_config["base_train_tasks"] += train_color_pair_tasks 

            train_shape_pair_tasks = [category_tasks.basic_rule("shape", ["triangle", "square"]), category_tasks.basic_rule("shape", ["triangle", "plus"]), category_tasks.basic_rule("shape", ["square", "plus"]), category_tasks.basic_rule("shape", ["square", "circle"]), category_tasks.basic_rule("shape", ["plus", "circle"]), category_tasks.basic_rule("shape", ["square", "emptysquare"]), category_tasks.basic_rule("shape", ["plus", "tee"]), category_tasks.basic_rule("shape", ["plus", "inverseplus"]), category_tasks.basic_rule("shape", ["circle", "emptysquare"]), category_tasks.basic_rule("shape", ["triangle", "inverseplus"])]
            run_config["base_train_tasks"] += train_shape_pair_tasks 

            if run_config["include_size_tasks"]:
                run_config["base_train_tasks"] += [category_tasks.basic_rule("size", ["16", "24"]), category_tasks.basic_rule("size", ["16", "32"])]

            # and eval tasks
            run_config["base_eval_tasks"] += [x for x in color_pair_tasks if x not in train_color_pair_tasks]
            run_config["base_eval_tasks"] += [category_tasks.basic_rule("shape", ["triangle", "circle"])] 
            if run_config["include_size_tasks"]:
                run_config["base_eval_tasks"] += [category_tasks.basic_rule("size", ["24", "32"])] 

        # now feature-conjunctive tasks, part random, with targeted holdouts 
        colors = category_tasks.BASE_COLORS.keys()
        shapes = category_tasks.BASE_SHAPES
        sizes = category_tasks.BASE_SIZES
        rules = ["AND", "OR", "XOR"]
        sc_composite_tasks = [category_tasks.composite_rule(
            r,
            category_tasks.basic_rule("shape", [s]),
            category_tasks.basic_rule("color", [c])) for s in shapes for c in colors for r in rules]

        if run_config["include_size_tasks"]:
            ssz_composite_tasks = [category_tasks.composite_rule(
                r,
                category_tasks.basic_rule("shape", [s]),
                category_tasks.basic_rule("size", [sz])) for s in shapes for sz in sizes for r in rules]

            csz_composite_tasks = [category_tasks.composite_rule(
                r,
                category_tasks.basic_rule("color", [c]),
                category_tasks.basic_rule("size", [sz])) for sz in sizes for c in colors for r in rules]

            composite_tasks = sc_composite_tasks + ssz_composite_tasks + csz_composite_tasks
        else:
            composite_tasks = sc_composite_tasks

        # some training examples for the held out mappings
        train_composite_tasks = [
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["triangle"]),
                category_tasks.basic_rule("color", ["yellow"])),
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["triangle"]),
                category_tasks.basic_rule("color", ["purple"])),
            category_tasks.composite_rule(
                "OR",
                category_tasks.basic_rule("shape", ["square"]),
                category_tasks.basic_rule("color", ["yellow"])),
            category_tasks.composite_rule(
                "OR",
                category_tasks.basic_rule("shape", ["square"]),
                category_tasks.basic_rule("color", ["purple"])),
#            category_tasks.composite_rule(
#                "XOR",
#                category_tasks.basic_rule("shape", ["emptytriangle"]),
#                category_tasks.basic_rule("color", ["yellow"])),
#            category_tasks.composite_rule(
#                "XOR",
#                category_tasks.basic_rule("shape", ["emptytriangle"]),
#                category_tasks.basic_rule("color", ["purple"])),
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["inverseplus"]),
                category_tasks.basic_rule("color", ["yellow"])),
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["inverseplus"]),
                category_tasks.basic_rule("color", ["purple"])),
#            category_tasks.composite_rule(
#                "OR",
#                category_tasks.basic_rule("shape", ["tee"]),
#                category_tasks.basic_rule("color", ["yellow"])),
#            category_tasks.composite_rule(
#                "OR",
#                category_tasks.basic_rule("shape", ["tee"]),
#                category_tasks.basic_rule("color", ["purple"])),
#            category_tasks.composite_rule(
#                "XOR",
#                category_tasks.basic_rule("shape", ["emptysquare"]),
#                category_tasks.basic_rule("color", ["yellow"])),
#            category_tasks.composite_rule(
#                "XOR",
#                category_tasks.basic_rule("shape", ["emptysquare"]),
#                category_tasks.basic_rule("color", ["purple"])),
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["plus"]),
                category_tasks.basic_rule("color", ["green"])),
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["circle"]),
                category_tasks.basic_rule("color", ["green"])),
#            category_tasks.composite_rule(
#                "OR",
#                category_tasks.basic_rule("shape", ["plus"]),
#                category_tasks.basic_rule("color", ["pink"])),
#            category_tasks.composite_rule(
#                "OR",
#                category_tasks.basic_rule("shape", ["circle"]),
#                category_tasks.basic_rule("color", ["pink"])),
#            category_tasks.composite_rule(
#                "XOR",
#                category_tasks.basic_rule("shape", ["plus"]),
#                category_tasks.basic_rule("color", ["cyan"])),
#            category_tasks.composite_rule(
#                "XOR",
#                category_tasks.basic_rule("shape", ["circle"]),
#                category_tasks.basic_rule("color", ["cyan"])),
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["plus"]),
                category_tasks.basic_rule("color", ["red"])),
            category_tasks.composite_rule(
                "AND",
                category_tasks.basic_rule("shape", ["circle"]),
                category_tasks.basic_rule("color", ["red"])),
#            category_tasks.composite_rule(
#                "OR",
#                category_tasks.basic_rule("shape", ["plus"]),
#                category_tasks.basic_rule("color", ["ocean"])),
#            category_tasks.composite_rule(
#                "OR",
#                category_tasks.basic_rule("shape", ["circle"]),
#                category_tasks.basic_rule("color", ["ocean"])),
            category_tasks.composite_rule(
                "XOR",
                category_tasks.basic_rule("shape", ["plus"]),
                category_tasks.basic_rule("color", ["blue"])),
            category_tasks.composite_rule(
                "XOR",
                category_tasks.basic_rule("shape", ["circle"]),
                category_tasks.basic_rule("color", ["blue"])),
            ]

        # and some eval
        train_composite_tasks += [
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
                category_tasks.basic_rule("color", ["purple"]))
            ]

        if run_config["include_size_tasks"]:
            train_composite_tasks += [
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
                category_tasks.composite_rule(
                    "AND",
                    category_tasks.basic_rule("shape", ["circle"]),
                    category_tasks.basic_rule("size", ["24"])),
                category_tasks.composite_rule(
                    "OR",
                    category_tasks.basic_rule("shape", ["square"]),
                    category_tasks.basic_rule("size", ["24"])),
                category_tasks.composite_rule(
                    "XOR",
                    category_tasks.basic_rule("shape", ["triangle"]),
                    category_tasks.basic_rule("size", ["24"])),
                category_tasks.composite_rule(
                    "AND",
                    category_tasks.basic_rule("color", ["red"]),
                    category_tasks.basic_rule("size", ["32"])),
                category_tasks.composite_rule(
                    "OR",
                    category_tasks.basic_rule("color", ["green"]),
                    category_tasks.basic_rule("size", ["32"])),
                category_tasks.composite_rule(
                    "OR",
                    category_tasks.basic_rule("color", ["yellow"]),
                    category_tasks.basic_rule("size", ["24"])),
                category_tasks.composite_rule(
                    "XOR",
                    category_tasks.basic_rule("color", ["ocean"]),
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
                category_tasks.basic_rule("color", ["purple"]))
            ]
        if run_config["include_size_tasks"]:
            eval_composite_tasks += [
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
        print(len(composite_tasks))

        train_composite_tasks += composite_tasks[:run_config["train_ext_composite_tasks"]]
        eval_composite_tasks += composite_tasks[run_config["train_ext_composite_tasks"]:]

        run_config["base_train_tasks"] += train_composite_tasks 
        run_config["base_eval_tasks"] += eval_composite_tasks

        # and some negations
#        train_negations = []
#        eval_negations = []
#        for x in basic_shape_tasks + basic_color_tasks + basic_size_tasks:
#            if x.accepted_list in [set(["purple"]), set(["circle"]), set([32])]:
#                eval_negations.append(category_tasks.negated(x))
#            else:
#                train_negations.append(category_tasks.negated(x))
#
#        negated_train_color_pair_tasks = [category_tasks.negated(x) for x in train_color_pair_tasks]
#        np.random.shuffle(negated_train_color_pair_tasks)
#        train_negations += negated_train_color_pair_tasks[:-5]
#        eval_negations += negated_train_color_pair_tasks[-5:]
#
#        negated_train_shape_pair_tasks = [category_tasks.negated(x) for x in train_shape_pair_tasks]
#        np.random.shuffle(negated_train_shape_pair_tasks)
#        train_negations += negated_train_shape_pair_tasks[:-2]
#        eval_negations += negated_train_shape_pair_tasks[-2:]
#
#        negated_train_composite_tasks = [category_tasks.negated(x) for x in train_composite_tasks]
#        np.random.shuffle(negated_train_composite_tasks)
#        train_negations += negated_train_composite_tasks[:-5]
#        eval_negations += negated_train_composite_tasks[-5:]
#
#        run_config["base_train_tasks"] += train_negations
#        run_config["base_eval_tasks"] += eval_negations

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

        # drop metamappings with too few training items
        to_remove = []
        for k,v in list(self.meta_pairings.items()):
            if len(v["train"]) < run_config["meta_min_train_threshold"]:
                del self.meta_pairings[k]
                to_remove.append(k)
        print(to_remove)

        self.meta_map_train_tasks = [x for x in self.meta_map_train_tasks if x not in to_remove]
        self.meta_map_eval_tasks = [x for x in self.meta_map_eval_tasks if x not in to_remove]

        print(len(self.meta_map_train_tasks))
        print(self.meta_map_eval_tasks)

        # and the base data points
        self.all_concept_instances = [category_tasks.categorization_instance(s, c, sz) for s in category_tasks.BASE_SHAPES for c in category_tasks.BASE_COLORS.keys() for sz in category_tasks.BASE_SIZES] 

        self.base_task_example_dicts = {str(t): category_tasks.construct_task_instance_dict(t, self.all_concept_instances) for t in self.base_train_tasks + self.base_eval_tasks}

    def fill_buffers(self, num_data_points=1):
        del num_data_points  # don't actually use in this version
        multiplicity = self.run_config["multiplicity"]
        memory_buffer_size = self.architecture_config["memory_buffer_size"]

        this_tasks = self.base_train_tasks + self.base_eval_tasks 
        for t in this_tasks:
            buff = self.memory_buffers[str(t)]
            x_data = np.zeros([self.architecture_config["memory_buffer_size"]] + self.architecture_config["input_shape"])
            y_data = np.zeros([self.architecture_config["memory_buffer_size"]] + self.architecture_config["output_shape"])
            examples = self.base_task_example_dicts[str(t)]
            index = 0
            for i, inst in enumerate(examples["positive"]):
                for j in range(multiplicity):
                    x_data[index, :, :, :] = inst.render()
                    index += 1
                    if index >= memory_buffer_size // 2:
                        break
                if index >= memory_buffer_size // 2:
                    break
            num_positive = index
            y_data[0:num_positive] = 1. 
            for i, contrasting in enumerate(examples["contrasting"]):
                these_contrasting = contrasting
                if len(these_contrasting) == 0:  # some examples don't have negative contrasts, e.g. in OR with both attributes 
                    these_contrasting = examples["all_negative"]
                    
                ex_perm = np.random.permutation(len(these_contrasting))
                for j in range(multiplicity):
                    x_data[index, :, :, :] = these_contrasting[ex_perm[j % len(ex_perm)]].render()
                    index += 1
                    if index >= memory_buffer_size:
                        break
                if index >= memory_buffer_size:
                    break

            if len(examples["other"]) > 0:
                ex_perm = np.random.permutation(len(examples["other"]))
                for i in range(self.architecture_config["memory_buffer_size"] - index):
                    x_data[index, :, :, :] = examples["other"][ex_perm[i % len(ex_perm)]].render()
                    index += 1
            else:  # for basic rules e.g. all examples "contrast" another
                pos_perm = np.random.permutation(len(examples["all_negative"]))
                pos_perm_len = len(pos_perm)
                for i in range(self.architecture_config["memory_buffer_size"] - index):
                    pos_ind = pos_perm[i % pos_perm_len]
                    x_data[index, :, :, :] = examples["all_negative"][pos_ind].render()
                    index += 1

            buff.insert(x_data, y_data, num_positive)

    def sample_from_memory_buffer(self, memory_buffer):
        """Return experiences from the memory buffer."""
        input_buff, output_buff, num_pos = memory_buffer.get_memories()
        return input_buff, output_buff, num_pos

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
            inputs, outputs, num_pos = self.sample_from_memory_buffer(memory_buffer)
            pos_indices = np.random.permutation(num_pos) 
            contr_neg_indices = pos_indices + num_pos  # matched contrasting indices

            small_set_size = min(num_pos, self.meta_batch_size//2)
            contr_set_size = small_set_size // 2 
            pos_indices = pos_indices[:small_set_size]

            if self.include_noncontrasting_negative:
                other_neg_indices = np.arange(
                    2*num_pos, self.architecture_config["memory_buffer_size"],
                    dtype=np.int32)
                np.random.shuffle(other_neg_indices)
                neg_indices = np.concatenate(
                    [contr_neg_indices[:contr_set_size],
                     other_neg_indices[:small_set_size - contr_set_size]], axis=0) 
            else:
                neg_indices = contr_neg_indices[:small_set_size]
            all_inds = np.concatenate([pos_indices, neg_indices], axis=0) 
        
            mask = np.zeros(len(all_inds), dtype=np.bool)

            feed_dict[self.base_input_ph] = inputs[all_inds]
            feed_dict[self.base_target_ph] = outputs[all_inds]

            # take half the pos
            mask[:small_set_size//2] = True
            if self.include_noncontrasting_negative:
                #  and half the neg (of each type) to be the meta-net samples
                mask[small_set_size:small_set_size + small_set_size//4] = True
                mask[-small_set_size//4:] = True
            else:
                #  and contrasting negative 
                mask[small_set_size:small_set_size + small_set_size//2] = True
            feed_dict[self.guess_input_mask_ph] = mask 
        else:   # meta dicts are the same
            return super(category_HoMM_model, self).build_feed_dict(
                task=task, lr=lr, fed_embedding=fed_embedding, call_type=original_call_type)

        if call_type == "fed":
            if len(fed_embedding.shape) == 1:
                fed_embedding = np.expand_dims(fed_embedding, axis=0)
            feed_dict[self.feed_embedding_ph] = fed_embedding
        elif call_type == "lang":
            feed_dict[self.language_input_ph] = self.task_name_to_lang_input[task_name]
        if call_type == "cached" or self.architecture_config["persistent_task_reps"]:
            feed_dict[self.task_index_ph] = [task_index]

        if call_type != "standard" and train_or_eval == "eval":
            feed_dict[self.guess_input_mask_ph] = np.zeros_like(feed_dict[self.guess_input_mask_ph])   # eval on all

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

    def get_new_memory_buffer(self):
        """Can be overriden by child"""
        return memory_buffer(length=self.architecture_config["memory_buffer_size"],
                             input_shape=self.architecture_config["input_shape"],
                             outcome_width=self.architecture_config["output_shape"][0])

    def _pre_loss_calls(self):
        def _logits_to_accuracy(x, labels=self.base_target_ph, 
                                backward_mask=self.guess_input_mask_ph):
            mask = tf.math.logical_not(backward_mask)  # only held out examples
            masked_x = tf.boolean_mask(x, mask)
            masked_labels = tf.boolean_mask(labels, mask)
            masked_vals = (1. + tf.math.sign(masked_x)) / 2.
            return tf.reduce_mean(tf.cast(tf.equal(masked_vals, masked_labels),
                                          tf.float32)) 
        self.base_accuracy = _logits_to_accuracy(self.base_output)
        self.base_fed_emb_accuracy =  _logits_to_accuracy(self.base_fed_emb_output)
        self.base_cached_emb_accuracy = _logits_to_accuracy(self.base_cached_emb_output)
        if self.run_config["train_language_base"]:
            self.base_lang_accuracy = _logits_to_accuracy(self.base_lang_output)

    def base_eval(self, task, train_or_eval):
        feed_dict = self.build_feed_dict(task, call_type="base_cached_eval")
        fetches = [self.total_base_cached_emb_loss, self.base_cached_emb_accuracy]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        name = str(task)
        return [name + "_loss:" + train_or_eval,
                name + "_accuracy:" + train_or_eval], res

    def base_language_eval(self, task, train_or_eval):
        feed_dict = self.build_feed_dict(task, call_type="base_lang_eval")
        fetches = [self.total_base_lang_loss, self.base_lang_accuracy]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        name = str(task)
        return [name + "_loss:" + train_or_eval,
                name + "_accuracy:" + train_or_eval], res

    def base_embedding_eval(self, embedding, task):
        feed_dict = self.build_feed_dict(task, fed_embedding=embedding, call_type="base_fed_eval")
        fetches = [self.base_fed_emb_accuracy]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        return res

    def intify_task(self, task_name):  # note: only base tasks implemented at present
        vocab_d = self.vocab_dict
        max_sentence_len = self.architecture_config["max_sentence_len"]

        underscore_split = task_name.split("_")
        if underscore_split[0] == "is":  #meta class
            full = [vocab_d[x] for x in underscore_split]
        elif underscore_split[0] == "switch":  # meta map 
            full = underscore_split[:2]
            for x in underscore_split[2:]:
                split_x = x.split("~")
                full += [split_x[0], "~", split_x[1]]
            full = [vocab_d[x] for x in full]
        else:  # basic task
            def intify_basic(basic_task_name):
                attribute_type, matches = basic_task_name.split("=")
                matches = matches.split("&") 
                full = [attribute_type, "="] + [x for m in matches for x in (m, "&")][:-1]
                return [vocab_d[x] for x in full]
            
            paren_split = re.split("[()]", task_name)
            if paren_split[0] in ["AND", "OR", "XOR"]:  # composite
                basic_1 = intify_basic(paren_split[2])
                basic_2 = intify_basic(paren_split[4])
                full = [vocab_d[paren_split[0]]] + [vocab_d["("]] * 2 + basic_1
                full += [vocab_d[")"], vocab_d["&"], vocab_d["("]] + basic_2
                full += [vocab_d[")"]] * 2
            else:
                full =  intify_basic(task_name)

        return [vocab_d["PAD"]] * (max_sentence_len - len(full)) + full


## running stuff
for run_i in range(run_config["run_offset"], run_config["run_offset"] + run_config["num_runs"]):
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    run_config["this_run"] = run_i

    model = category_HoMM_model(run_config=run_config)
    model.run_training()

    tf.reset_default_graph()
