import matplotlib.pyplot as plt


def get_data_list(file_path):
    step_list, time_list, acc_list = [], [], []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                line = line.split()
                step, time, acc = int(line[0]), float(line[1]), float(line[2])
                step_list += [step]
                time_list += [time]
                acc_list += [acc]
            except Exception as e:
                # print(file_path, e)
                pass
    return (step_list, time_list, acc_list)


file_names = (
    "slide_256_Amazon670K.txt",
    "PyTorchImplementation.txt",
    "tf_gpu_256_Amazon670K.txt",
    "tf_cpu_256_Amazon670K.txt",
)

slide_data, pt_data, tf_gpu_data, tf_cpu_data = map(get_data_list, file_names)

n_data = []

# Accuracy vs Time plot
plt.xscale("log")
plt.plot(slide_data[1], slide_data[2], label="SLIDE")
plt.plot(tf_gpu_data[1], tf_gpu_data[2], label= "TF-GPU")
plt.plot(tf_cpu_data[1], tf_cpu_data[2], label="TF-CPU")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Test accuracy")
plt.title("SLIDE vs TF-GPU and TF-CPU")
plt.tight_layout()
plt.savefig("SLIDEvsTF-GPU.png")
