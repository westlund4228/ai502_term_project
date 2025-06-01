import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

for model in df['model'].unique():
    plt.figure(figsize=(6, 4))
    for method in df['method'].unique():
        for stage in ['ft', 'search']:
            sub = df[(df['model'] == model) & (df['method'] == method) & (df['stage'] == stage)]
            label = f"{method}-{stage}"
            plt.plot(sub['flops_ratio'], sub['acc'], marker='o', label=label)
    if model == 'mobilenet':
        model_name = 'MobileNet'
    elif model == 'resnet':
        model_name = 'ResNet'
    else:
        raise NameError('Incorrect model name')
    plt.title(f"{model_name}: Acc vs. FLOPs ratio")
    plt.xlabel("FLOPs (%)")
    plt.ylabel("Top1 Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model}_acc_flops.png")
    # plt.show()
