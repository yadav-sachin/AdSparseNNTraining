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
                print(file_path, e)
                pass
    return (step_list, time_list, acc_list)


file_names = (
    "slide_256_Amazon670K.txt",
    # "slide2.txt",
    "PyTorchImplementation.txt",
    "tf_gpu_256_Amazon670K.txt",
    "tf_cpu_256_Amazon670K.txt",
    # "my.txt"
)

print("1")
# slide_data, pt_data, tf_gpu_data, tf_cpu_data, my_data = map(get_data_list, file_names)
slide_data, pt_data, tf_gpu_data, tf_cpu_data = map(get_data_list, file_names)

n_data = []
# for i in range(len(my_data)):
#     if i % 50:
#         n_data.append(my_data[i])
        
# print(pt_data)
print("2")
# Accuracy vs Time plot
plt.xscale("log")
plt.plot(slide_data[1], slide_data[2], label="SLIDE")
plt.plot(tf_gpu_data[1], tf_gpu_data[2], label= "TF-GPU")
# plt.plot(pt_data[1], pt_data[2], label="PyTorch GPU baseline")
print(min(slide_data[1]))
print(min(pt_data[1]))
plt.plot(tf_cpu_data[1], tf_cpu_data[2], label="TF-CPU")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Test accuracy")
# plt.plot(my_data[1], my_data[2])
plt.title("SLIDE vs PyTorch GPU baseline")
plt.tight_layout()
plt.savefig("h.png")
