import math
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["figure.autolayout"] = True

# Plotting settings
data_file = "./results.csv"
font_scale = 5
fig_size = (50, 30)
grouped_bar_order = ["Baseline", "MII-Public", "MII-Azure"]
grouped_bar_color = [
    "orange",
    "green",
    "blue",
]  # Should align with order of grouped_bar_order
overhead_color = grouped_bar_color
overhead_legend_color = "gray"
overhead_hatch = "///"
x_label = "Model"
y_label = "Latency (ms)"

# Model names we want to keep for x-axis ticks
keep_model_names = [
    "bert-large-uncased",
    "bert-base-cased",
    "bert-based-uncased",
    "gpt2-xl",
    "gpt2-large",
    "gpt2-medium",
    "gpt2",
    "distilgpt2",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-j-6B",
    "roberta-large",
    "roberta-base",
    "distilroberta-base",
    "facebook/opt-2.7b",
    "facebook/opt-1.3b",
    "facebook/opt-350m",
    "facebook/opt-125m",
    "bigscience/bloom-7b1",
    "bigscience/bloom-3b",
    "bigscience/bloom-1b7",
    "bigscience/bloom-1b1",
    "bigscience/bloom-560m",
]

# When plotting models in keep_model_names, remove certain prefixes
prefix_to_remove = ["facebook/", "bigscience/", "EleutherAI/"]

# How we want to rename other models ("task" : "name prefix")
rename_model_dict = {
    "text-generation": "text-gen",
    "fill-mask": "fill-mask",
    "token-classification": "token-class",
    "text-classification": "text-class",
    "question-answering": "q&a",
}

sns.set(style="ticks", font_scale=font_scale)

# Read the data
data = pd.read_csv(data_file)

# Combine all GPT style models into single plot
data["model family"] = data["model family"].apply(lambda x: "gpt" if "gpt" in x else x)

# Plot by model family
for model_family in data["model family"].unique():
    # Get just the data for this model family
    family_data = data[data["model family"] == model_family]

    # Get model order, by average e2e latency
    idx = family_data["setup"] == "baseline"
    model_order = family_data[idx].sort_values("e2e latency Avg",
                                               ascending=False)["model"].values

    # Create a new figure and axis
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # Plot the e2e time
    e2e_bar = sns.barplot(
        data=family_data,
        x="model",
        y="e2e latency Avg",
        hue="branch",
        palette=overhead_color,
        ax=ax,
        width=0.5,
        hue_order=grouped_bar_order,
        order=model_order,
    )

    # Add hatch to overhead (i.e., e2e plot)
    # NOTE: MUST BE DONE BEFORE NEXT PLOT ADDED
    for bar in e2e_bar.patches:
        bar.set_hatch(overhead_hatch)

    # Overlay the model time
    forward_bar = sns.barplot(
        data=family_data,
        x="model",
        y="model latency Avg",
        palette=grouped_bar_color,
        hue="branch",
        hue_order=grouped_bar_order,
        ax=ax,
        width=0.5,
        order=model_order,
    )

    # Add horizontal grid lines and set yticks
    y_max = math.ceil(ax.get_ylim()[-1])
    plt.yticks([i for i in range(0, y_max, 2)])
    plt.grid(axis="y")

    # Adjust the legend so we don't have duplicate labels, and add overhead label
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [Patch(facecolor=overhead_legend_color,
               hatch=overhead_hatch)] + handles[len(grouped_bar_order):],
        ["Pre/post processing"] + labels[len(grouped_bar_order):],
        title=None,
        #bbox_to_anchor=(1.02, 0.15),
        loc="upper right",
        ncol=2,
    )
    """
    # Rotate the x-axis tick labels so they are readable
    # Also replace them with coded names (where appropriate)
    labels = []
    make_bold = []
    m_count = {}
    alias_dict = {}
    for idx, row in ordered_data.iterrows():
        #print(f"Processing {row['model-family']} {row['model']}")
        if True:  #row["model-family"] == "bert":
            task = row["task"]
            model_name = row["model"]
            if model_name in keep_model_names:
                for prefix in prefix_to_remove:
                    model_name = model_name.replace(prefix, "")
                labels.append(model_name)
                make_bold.append(True)
            else:
                if task not in m_count:
                    m_count[task] = 1
                if row["model-family"] == "bert":
                    alias = f"bert-{rename_model_dict[task]}-m{m_count[task]}"
                else:
                    alias = f"{rename_model_dict[task]}-m{m_count[task]}"
                alias_dict[alias] = model_name
                labels.append(alias)
                make_bold.append(False)
                m_count[task] += 1
    plt.xticks(labels=labels, ticks=range(len(labels)), rotation=45, ha="right")

    # When we have model aliases, let's highlight the model names that can be recognized
    # if not all(make_bold):
    #    for idx, label in enumerate(ax.get_xticklabels()):
    #        if make_bold[idx]:
    #            label.set_fontweight("bold")

    # Set x- and y-axis labels
    plt.xlabel("")
    plt.ylabel(y_label)
    plt.savefig(f"./{model_family}.png")

    # print model aliases mapping
    for alias, model_name in alias_dict.items():
        print(f"| {alias: >14} | {model_name} |")
    print("-------------------------------------")
    #exit(0)

    # save ordered_data to csv file
    #print(f"ordered_data for {model_family} is {model_family_df}")

    #model_family_df.to_csv(f"./{model_family}.csv", index=False)
    #model_family_df.pivot(index='model', columns='setup', values='e2e latency').reset_index(drop=False).to_csv(f'{model_family}.csv')
    model_family_df.pivot(index='model',
                          columns='setup',
                          values='e2e latency').reset_index(drop=False).reindex(
                              ["model",
                               "Baseline",
                               "MII-Public",
                               "MII-Azure"],
                              axis=1).to_csv(f'{model_family}.csv',
                                             index=False)

all_data_df.pivot(index='model',
                  columns='setup',
                  values='e2e latency').reset_index(drop=False).reindex(
                      ["model",
                       "Baseline",
                       "MII-Public",
                       "MII-Azure"],
                      axis=1).to_csv(f'All.csv',
                                     index=False)

for idx, row in all_data_df.iterrows():
    #print(f'model = {row["model"]}')
    if len(row["model"].split("/")) > 1:
        row["model"] = row["model"].split("/")[
            1]  # keep the second part of the model name
    else:
        row["model"] = row["model"]

    if len(row["model"]) > 25:
        # drop this row
        all_data_df.drop(idx, inplace=True)

    pass  # print(f'model = {row["model"]}')

#for idx, row in all_data_df.iterrows():
# rename this row
#row["model"] = row["model"][:25] + "..."
#model_name.replace(prefix, "")

#print(f'all_data_df = {all_data_df}')
#for idx, row in all_data_df.iterrows():
#    print(f"model = {row['model']}, idx = {idx}")
"""
