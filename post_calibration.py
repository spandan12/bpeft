import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_ece(
    max_probs, 
    true_classes,
    predicted_classes, 
    n_bins=50):

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        in_bin = np.where((max_probs > bin_lower) & (max_probs <= bin_upper))[0]
        if len(in_bin) > 0:
            accuracy = np.mean(predicted_classes[in_bin] == true_classes[in_bin])
            avg_confidence = np.mean(max_probs[in_bin])
            temp_ece = np.abs(avg_confidence - accuracy) * len(in_bin) / len(predicted_classes)
            ece += temp_ece
            
    return ece


def plot_reliability_diagrams(save_loc, title, true_classes, predicted_probs, predicted_classes,n_bins = 50):
    bins = np.linspace(0, 1, n_bins)
    Bm = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_counts = []
    for m in range(n_bins):
        a, b = m/n_bins, (m+1)/n_bins
        for i in range(len(true_classes)):
            if predicted_probs[i]>a and predicted_probs[i]<=b:
                Bm[m]+=1
                if true_classes[i]==predicted_classes[i]:
                    bin_acc[m]+=1
                bin_conf[m]+=predicted_probs[i]
        if Bm[m]!=0:
            bin_acc[m] = bin_acc[m]/Bm[m]
            bin_conf[m] = bin_conf[m]/Bm[m]
    
    bin_to_plot = []
    for index, bin_co in enumerate(bin_conf):
        if bin_co == 0:
            bin_to_plot.append(0)
        else:
            bin_to_plot.append(bins[index])

    histogram_to_plot = [bin_co for bin_co in bin_conf]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    plt.title(title, size=30)
    plt.ylabel("Accuracy",  size=30)
    plt.xlabel("Confidence",  size=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_axisbelow(True)
    ax.grid(color = 'gray', linestyle = 'dashed')
    plt.bar(bins, bin_to_plot, width = 0.1, alpha = 0.3, edgecolor = 'black', color = 'r', hatch = '\\')
    plt.bar(bins, bin_acc, width = 0.1, alpha = 0.1, edgecolor = 'black', color = 'b')
    plt.plot([0, 1], [0, 1], '--', color = 'gray', linewidth = 2)
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.show()
    plt.savefig(save_loc, bbox_inches = 'tight')
    plt.clf()
    plt.close()

base_path = "./logs/evidential_1_FSTR_cifar10_kl_0.1/seed1/cifar10/sup_vitb16_imagenet21k/lr0.1_wd0.01/run1"
data_in_path = f'{base_path}/_full_matrix_seed.csv'
m_value = 1.5

df_in = pd.read_csv(data_in_path, index_col=False)

IN_data = df_in.to_numpy()

predicted_value = IN_data[:, 0]
gt_value = IN_data[:, 1]
vacuity=IN_data[:, 2]
evidence = IN_data[:, 4:] 
num_size = evidence.shape[0]
min_evidence = np.min(evidence,1)
alpha = evidence + (1)
accuracy = np.sum(predicted_value==gt_value)/(gt_value.shape[0])*100
row_sums_plus1 = np.sum(alpha, axis=1, keepdims=True)

# Step 3: Normalize each row by its respective sum
normalized_evidence = alpha / row_sums_plus1

max_values = np.max(normalized_evidence, axis=1, keepdims=True)  # Shape: (10000, 1)

ece = calculate_ece(
    max_probs=max_values, 
    true_classes=gt_value,
    predicted_classes=predicted_value, 
    n_bins=10)

plot_reliability_diagrams(save_loc=f'{base_path}/pre_ece.png', title = f'Accuracy: {accuracy:.3f} ECE: {ece:.3f}',
                          true_classes = gt_value,
                          predicted_probs = max_values, 
                          predicted_classes = predicted_value,
                          n_bins = 10)

print(f'Before BPEFT, Accuracy: {accuracy:.3f} ECE: {ece:.3f}')
print(f'Reliability plot saved in {base_path}/pre_ece.png')

min_evidence = np.min(evidence,1)
numerator = evidence - min_evidence.reshape(num_size,1)
denominator = min_evidence.reshape(num_size,1)
base_rate = (numerator/denominator)**(m_value)
alpha = evidence + (base_rate * 100)

row_sums_plus1 = np.sum(alpha, axis=1, keepdims=True)

# Step 3: Normalize each row by its respective sum
normalized_evidence = alpha / row_sums_plus1

# Get the maximum value of each row
max_values = np.max(normalized_evidence, axis=1, keepdims=True)  # Shape: (10000, 1)





ece = calculate_ece(
    max_probs=max_values, 
    true_classes=gt_value,
    predicted_classes=predicted_value, 
    n_bins=10)

plot_reliability_diagrams(save_loc=f'{base_path}/post_ece.png', title = f'Accuracy: {accuracy:.3f} ECE: {ece:.3f}',
                          true_classes = gt_value,
                          predicted_probs = max_values, 
                          predicted_classes = predicted_value,
                          n_bins = 10)

print(f'After BPEFT, Accuracy: {accuracy:.3f} ECE: {ece:.3f}')
print(f'Reliability plot saved in {base_path}/post_ece.png')