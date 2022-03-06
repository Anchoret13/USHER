import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

ci = lambda x, z=2: (np.mean(x) - z*np.std(x)/len(x)**.5, np.mean(x) + z*np.std(x)/len(x)**.5 )
err_bar = lambda x, z=2: z*np.std(x)/len(x)**.5
keylist = ["success_rate", "average_reward", "average_initial_value"]
display_keylist = ["bias"]
#loc = "train_her_mod_logging"
loc = "logging"

def format_name(inpt): 	
	return ' '.join([s.capitalize() for s in inpt.split("_")])

def format_title(inpt): 	
	return (inpt.replace("RandomGridworld", " Random Obstacles")
			.replace("Gridworld", " Continuous Long/Short Path Environment")
			.replace("Alt ", ""))

def format_method(inpt):
	if "1-goal" in inpt: 
		return "HER"
	elif "2-goal ratio" in inpt:
		return "USHER (proposed)"
	elif "2-goal" in inpt:
		return "2-goal HER"
	assert False

def parse_log(env_name):
	filename = f"{loc}/{env_name}.txt"
	with open(filename, "r") as f: 
		file = f.read()

	file_list = file.split("\n")
	experiment_dict = {}
	# current_list = []
	current_dict = {key: [] for key in keylist}
	for line in reversed(file_list):
		if line == '': 
			continue
		elif "run" in line: 
			env, ratio_clip_string, method = tuple(line.split(","))
			ratio_clip = float(ratio_clip_string[:-len("ratio_clip")])
			# if method not in experiment_dict.keys():
			# 	experiment_dict[method] = {key: [] for key in keylist}

			if method not in experiment_dict.keys():
				experiment_dict[method] = {key: [] for key in keylist}
				experiment_dict[method]["bias"] = []
			for key in keylist:
				# if not len(current_dict[key]) > 0:
				mean = np.mean(current_dict[key])
				list_ci = ci(current_dict[key])
				experiment_dict[method][key].append((ratio_clip, mean, list_ci))

			bias = [v - r for v, r in zip(current_dict["average_initial_value"], current_dict["average_reward"])]
			mean = np.mean(bias)
			list_ci = ci(bias)
			experiment_dict[method]["bias"].append((ratio_clip, mean, list_ci))
			current_dict = {key: [] for key in keylist}
		else: 
			# data = line.split(",")
			# num = 0#float(line.split(":")[-1])
			# for item in data: 
				# if "success_rate" in item: 
				# 	num = float(line.split(":")[-1])
			for key in keylist:
				if key in line: 
					current_dict[key].append(float(line.split(":")[-1]))
			# current_list.append(num)

	return experiment_dict

def plot_log(experiment_dict, name="Environment"):
	method = [i for i in experiment_dict.keys()][0]
	if len(experiment_dict[method][display_keylist[0]]) > 1:
		line_plot(experiment_dict, name=name)
	else: 
		bar_plot(experiment_dict, name=name)

def line_plot(experiment_dict, name="Environment"):
	color_list = ["red", "green", "blue", "brown", "pink"]
	for key in display_keylist + keylist:
		i=0
		for method in experiment_dict.keys():
			ratio_clip_list = [elem[0] for elem in experiment_dict[method][key]]
			mean_list = [elem[1] for elem in experiment_dict[method][key]]
			upper_ci_list = [elem[2][0] for elem in experiment_dict[method][key]]
			lower_ci_list = [elem[2][1] for elem in experiment_dict[method][key]]

			plt.plot(ratio_clip_list, mean_list, color=color_list[i],label=format_method(method))
			plt.fill_between(ratio_clip_list, lower_ci_list, upper_ci_list, color=color_list[i], alpha=.1)

			i += 1

		plt.xlabel("ratio_clip")
		plt.ylabel(format_name(key))
		env_title = format_title(name)
		plt.title(f"{env_title} Performance")
		plt.legend()
		plt.savefig(f"{loc}/images/{name}__{key}.png")
		plt.show()

	# pdb.set_trace()

def bar_plot(experiment_dict, name="Environment"):
	color_list = ["red", "green", "blue", "brown", "pink"]
	methods = [m for m in experiment_dict.keys()]
	ticks = list(range(len(methods)))
	for key in keylist:
		i=0
		# for method in experiment_dict.keys():
		# 	mean_list = [elem[1] for elem in experiment_dict[method][key]]
		# 	upper_ci_list = [elem[2][0] for elem in experiment_dict[method][key]]
		# 	lower_ci_list = [elem[2][1] for elem in experiment_dict[method][key]]

		mean_list = [experiment_dict[method][key][0][1] for method in methods]
		var_list = [experiment_dict[method][key][0][2][0] - experiment_dict[method][key][0][2][1] for method in methods]

		plt.bar(ticks, mean_list, yerr=var_list)
		plt.xticks(ticks, experiment_dict.keys())


		# plt.xlabel("ratio_clip (fraction of maximum action)")
		plt.ylabel(format_name(key))
		plt.title(f"{name} Performance")
		plt.legend()
		plt.savefig(f"{loc}/images/{name}__{key}.png")
		plt.show()



if __name__ == '__main__':
	# env = "RandomGridworld"
	# env = "2Dnav"
	envs = sys.argv
	if len(envs) == 1: 
		print("No environent given")
	else: 
		[plot_log(parse_log(env), name=env) for env in envs[1:]]
# 	print(plot_log(parse_log("logging/Asteroids.txt")))
	# print(plot_log(parse_log(env), name=env))
