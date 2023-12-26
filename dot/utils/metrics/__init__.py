def save_metrics(metrics, path):
    names = list(metrics.keys())
    num_values = len(metrics[names[0]])
    with open(path, "w") as f:
        f.write(",".join(names) + "\n")
        for i in range(num_values):
            f.write(",".join([str(metrics[name][i]) for name in names]) + "\n")