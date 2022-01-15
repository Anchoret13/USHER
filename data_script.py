import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ci = lambda x, z=2: (np.mean(x) - z*np.std(x)/len(x)**.5, np.mean(x) + z*np.std(x)/len(x)**.5 )
err_bar = lambda x, z=2: z*np.std(x)/len(x)**.5

# float_vals = list(map(float, [".01", ".03", ".1"]))

goals = ["one_goal", "two_goal"]
# vals = ["0", ".01", ".03", ".1"]
vals = [".01", ".03", ".1"]
data_dict = {key1:{key2:None for key2 in vals} for key1 in goals}

data_dict["one_goal"][".1"] = [.296, .429, .379, .308, .4, .458, .412, .408, 
	.450, .146, .312, .425, .404, .333, .4, .458, .412, .113, .233, .392]
data_dict["two_goal"][".1"] = [.433, .450, .45, .404, .487, .438, .429, .425, 
	.454, .433, .379, .417, .438, .429, .467, .412, .383, .421, .438, .463]

data_dict["one_goal"][".01"] = [.6, .554, .529, .404, .304, .442, .513, .629, 
	.525, .504, .479, .25, .142, .038, .212, .296, .379, .438, .495, .554]
data_dict["two_goal"][".01"] = [.617, .604, .467, .575, .65, .667, .633, .679,
	.679, .629, .683, .550, .558, .667, .550, .679, .650, .683, .646, .729]

data_dict["one_goal"][".03"] = [.562, .546, .329, .317, .500, .542, .558, .479, 
	.637, .562, .487, .508, .504, .600, .575, .604, .558, .567, .663, .600]
data_dict["two_goal"][".03"] = [.633, .554, .525, .621, .575, .587, .612, .517, 
	.600, .537, .450, .583, .587, .571, .583, .658, .517, .625, .604, .617]
 
# for key2 in [".01", ".03", ".1"]:
# 	for key1 in ["one_goal", "two_goal"]:

combinations = ((key1, key2) for key1 in goals for key2 in vals)
for (key1, key2) in combinations:
	mean = np.mean(data_dict[key1][key2])
	ci_tuple = ci(data_dict[key1][key2])
	err_bar_tuple = err_bar(data_dict[key1][key2])
	print(f"[{key1}][{key2}]   \tCI: {ci_tuple}")

means = {key1: {key2: np.mean(data_dict[key1][key2]) for key2 in vals} for key1 in goals}
cis = {key1: {key2: ci(data_dict[key1][key2]) for key2 in vals} for key1 in goals}

float_vals = list(map(float, vals))

lower_bound = {key1: [cis[key1][key2][0] for key2 in vals] for key1 in goals}
upper_bound = {key1: [cis[key1][key2][1] for key2 in vals] for key1 in goals}

goal_colors = {"one_goal": "r", "two_goal": "b"}

plt.plot(float_vals, [means["one_goal"][key2] for key2 in vals], color=goal_colors["one_goal"],label='one goal')
plt.plot(float_vals, [means["two_goal"][key2] for key2 in vals], color=goal_colors["two_goal"], label='two goals (proposed)')

for goal in goals:
	plt.fill_between(float_vals, lower_bound[goal], upper_bound[goal], \
		color=goal_colors[goal], alpha=.1)


plt.xlabel("Noise (fraction of maximum action)")
plt.ylabel("Success Rate")
plt.title("FetchSlide Performance")
plt.legend()
plt.show()
